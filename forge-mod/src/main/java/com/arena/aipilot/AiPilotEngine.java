package com.arena.aipilot;

import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.player.ClientPlayerEntity;
import net.minecraft.block.Blocks;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.ai.attributes.AttributeModifier;
import net.minecraft.entity.ai.attributes.Attributes;
import net.minecraft.entity.item.ArmorStandEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.inventory.EquipmentSlotType;
import net.minecraft.entity.item.EnderCrystalEntity;
import net.minecraft.item.AxeItem;
import net.minecraft.item.BlockItem;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;
import net.minecraft.item.ShieldItem;
import net.minecraft.item.SplashPotionItem;
import net.minecraft.item.SwordItem;
import net.minecraft.item.TridentItem;
import net.minecraft.potion.Effect;
import net.minecraft.potion.EffectInstance;
import net.minecraft.potion.Effects;
import net.minecraft.potion.PotionUtils;
import net.minecraft.util.Direction;
import net.minecraft.util.Hand;
import net.minecraft.util.MovementInput;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.BlockRayTraceResult;
import net.minecraft.util.math.vector.Vector3d;
import net.minecraftforge.client.event.RenderGameOverlayEvent;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Логика авто-боя. Ничего не отправляет в сеть напрямую — только вращает камеру,
 * жмёт инпуты и вызывает ванильные действия клиента (атака, использование предмета,
 * смена слота). Для сервера это неотличимо от обычного игрока.
 */
public final class AiPilotEngine {

    private static double searchRange = 17.0;
    private static LivingEntity target;
    private static final PpoPolicy ppoPolicy = new PpoPolicy();
    private static final float[] lastObservation = new float[34];
    private static final int[] actionCooldownUntilTick = new int[30];
    private static int lastAction = -1;
    private static int modelStepCount = 0;
    private static int observedTargetId = -1;
    private static int combatWeaponSlot = -1;
    private static float previousHealth = Float.NaN;
    private static float previousTargetHealth = Float.NaN;
    private static int playerComboTaken = 0;
    private static int enemyComboTaken = 0;
    private static int lastPlayerDamageTick = Integer.MIN_VALUE / 2;
    private static int lastEnemyDamageTick = Integer.MIN_VALUE / 2;
    private static int lastMeleeAttackTick = -1_000_000;
    private static int counterUntilTick = 0;
    private static int counterArmedUntilTick = 0;
    private static int enemyStunUntilTick = 0;
    private static int crystalDetonateUntilTick = 0;
    private static int rangedReleaseTick = 0;
    private static BlockPos placedCrystalBase = null;
    private static boolean shieldActionActive = false;
    private static int lastPolicyTick = Integer.MIN_VALUE;

    // Интенты движения, которые выставляет «мозг» и применяет InputUpdateEvent.
    private static float inForward = 0.0F;
    private static float inStrafe = 0.0F;
    private static boolean inJump = false;
    private static boolean inSprint = false;

    private AiPilotEngine() {}

    public static void reset() {
        ClientPlayerEntity localPlayer = Minecraft.getInstance().player;
        if (shieldActionActive && localPlayer != null) localPlayer.stopUsingItem();
        target = null;
        Arrays.fill(lastObservation, 0.0F);
        Arrays.fill(actionCooldownUntilTick, 0);
        lastAction = -1;
        modelStepCount = 0;
        observedTargetId = -1;
        combatWeaponSlot = -1;
        previousHealth = Float.NaN;
        previousTargetHealth = Float.NaN;
        playerComboTaken = 0;
        enemyComboTaken = 0;
        lastPlayerDamageTick = Integer.MIN_VALUE / 2;
        lastEnemyDamageTick = Integer.MIN_VALUE / 2;
        lastMeleeAttackTick = -1_000_000;
        counterUntilTick = 0;
        counterArmedUntilTick = 0;
        enemyStunUntilTick = 0;
        crystalDetonateUntilTick = 0;
        rangedReleaseTick = 0;
        placedCrystalBase = null;
        shieldActionActive = false;
        lastPolicyTick = Integer.MIN_VALUE;
        inForward = 0.0F;
        inStrafe = 0.0F;
        inJump = false;
        inSprint = false;
    }

    public static void setRange(double blocks) {
        searchRange = blocks;
        AiPilot.chat("\u00A7bAI Pilot: радиус поиска " + String.format(Locale.ROOT, "%.1f", blocks));
    }

    // ------------------------------------------------------------------ PPO brain
    /** Runs the trained deterministic actor at the training cadence: ten decisions per second. */
    public static void tick(Minecraft mc) {
        ClientPlayerEntity p = mc.player;
        if (p == null || mc.level == null || mc.gameMode == null) return;
        if (mc.screen != null || p.isDeadOrDying()) {
            releaseKeys(mc);
            return;
        }
        if (!ppoPolicy.isReady()) {
            // Deliberately fail closed: never silently present the legacy heuristic as PPO.
            releaseKeys(mc);
            lastAction = -1;
            return;
        }
        releaseChargedRangedItem(mc, p);

        if (target == null || !target.isAlive() || target.level != p.level
                || horizontalDistance(p, target) > searchRange
                || (target instanceof PlayerEntity && (((PlayerEntity) target).isCreative()
                    || ((PlayerEntity) target).isSpectator()))) {
            target = findTarget(mc, p);
        }
        if (target == null) {
            releaseKeys(mc);
            observedTargetId = -1;
            return;
        }

        if (observedTargetId != target.getId()) {
            resetEpisodeState();
            observedTargetId = target.getId();
            if (!p.isUsingItem()) {
                selectBestWeapon(p);
                combatWeaponSlot = p.inventory.selected;
            }
        }
        if (combatWeaponSlot < 0 && !p.isUsingItem()) {
            selectBestWeapon(p);
            combatWeaponSlot = p.inventory.selected;
        }
        if (!p.isUsingItem() && combatWeaponSlot >= 0 && p.inventory.selected != combatWeaponSlot) {
            p.inventory.selected = combatWeaponSlot;
        }
        double dist = horizontalDistance(p, target);
        aimAt(p, target, dist);
        updateComboState(p, target);
        detonatePlacedCrystal(mc, p);

        // Training DT is 0.10s. Forge ticks are 0.05s, so infer once every two ticks.
        if ((p.tickCount & 1) == 0 && lastPolicyTick != p.tickCount) {
            lastPolicyTick = p.tickCount;
            buildObservation(p, target, mc);
            lastAction = ppoPolicy.predict(lastObservation);
            modelStepCount = Math.min(600, modelStepCount + 1);
            applyPolicyAction(mc, p, target, lastAction);
        }
    }

    private static void resetEpisodeState() {
        Arrays.fill(actionCooldownUntilTick, 0);
        modelStepCount = 0;
        previousHealth = Float.NaN;
        previousTargetHealth = Float.NaN;
        playerComboTaken = 0;
        enemyComboTaken = 0;
        lastPlayerDamageTick = Integer.MIN_VALUE / 2;
        lastEnemyDamageTick = Integer.MIN_VALUE / 2;
        lastMeleeAttackTick = -1_000_000;
        counterUntilTick = 0;
        counterArmedUntilTick = 0;
        enemyStunUntilTick = 0;
        crystalDetonateUntilTick = 0;
        rangedReleaseTick = 0;
        placedCrystalBase = null;
        lastAction = -1;
        lastPolicyTick = Integer.MIN_VALUE;
        Arrays.fill(lastObservation, 0.0F);
    }

    private static void buildObservation(ClientPlayerEntity p, LivingEntity t, Minecraft mc) {
        Arrays.fill(lastObservation, 0.0F);
        double dx = t.getX() - p.getX();
        double dz = t.getZ() - p.getZ();
        double dist = Math.sqrt(dx * dx + dz * dz);
        PlayerEntity opponent = t instanceof PlayerEntity ? (PlayerEntity) t : null;
        Vector3d pv = p.getDeltaMovement();
        Vector3d tv = t.getDeltaMovement();
        int tick = p.tickCount;

        lastObservation[0] = clip((p.getHealth() + p.getAbsorptionAmount()) / 20.0F, 0.0F, 1.5F);
        lastObservation[1] = clip((t.getHealth() + t.getAbsorptionAmount()) / 20.0F, 0.0F, 1.5F);
        lastObservation[2] = clip((float) (dist / 20.0), 0.0F, 1.0F);
        lastObservation[3] = clip((float) Math.sqrt(pv.x * pv.x + pv.z * pv.z), 0.0F, 2.0F);
        lastObservation[4] = clip((float) Math.sqrt(tv.x * tv.x + tv.z * tv.z), 0.0F, 2.0F);
        lastObservation[5] = clip((float) (dx / 20.0), -1.0F, 1.0F);
        lastObservation[6] = clip((float) (dz / 20.0), -1.0F, 1.0F);
        lastObservation[7] = p.isOnGround() ? 1.0F : 0.0F;
        lastObservation[8] = tick - lastMeleeAttackTick
                >= weaponCooldown(p, combatWeaponSlot) * 20.0F ? 1.0F : 0.0F;
        lastObservation[9] = clip(modelStepCount / 600.0F, 0.0F, 1.0F);
        lastObservation[10] = clip(weaponDamage(p, combatWeaponSlot), 0.0F, 14.0F) / 14.0F;
        lastObservation[11] = clip(weaponCooldown(p, combatWeaponSlot) / 1.2F, 0.0F, 1.0F);
        lastObservation[12] = clip(weaponDamage(t, -1), 0.0F, 14.0F) / 14.0F;
        lastObservation[13] = armorTierReduction(p.getArmorValue());
        lastObservation[14] = armorTierReduction(t.getArmorValue());
        lastObservation[15] = clip(countHealingApples(p) / 5.0F, 0.0F, 1.0F);
        lastObservation[16] = clip(countSplashPotions(p, Effects.HEAL) / 5.0F, 0.0F, 1.0F);
        lastObservation[17] = clip(countInventoryItem(p, Items.ENDER_PEARL) / 4.0F, 0.0F, 1.0F);
        int buffs = countPotionsWithEffect(p, Effects.DAMAGE_BOOST)
                + countPotionsWithEffect(p, Effects.MOVEMENT_SPEED);
        int ranged = countInventoryItem(p, Items.SNOWBALL) + countInventoryItem(p, Items.EGG)
                + countInventoryItem(p, Items.ARROW) + countInventoryItem(p, Items.TIPPED_ARROW)
                + countInventoryItem(p, Items.SPECTRAL_ARROW) + countInventoryItem(p, Items.CROSSBOW);
        lastObservation[18] = clip((buffs + ranged) / 12.0F, 0.0F, 1.0F);
        lastObservation[19] = isShieldBlocking(p) ? 1.0F : 0.0F;
        lastObservation[20] = isShieldBlocking(t) ? 1.0F : 0.0F;
        lastObservation[21] = clip(countBuildingBlocks(p) / 24.0F, 0.0F, 1.0F);
        int utilities = countInventoryItem(p, Items.COBWEB) + countInventoryItem(p, Items.LAVA_BUCKET)
                + countInventoryItem(p, Items.WATER_BUCKET);
        lastObservation[22] = clip(utilities / 6.0F, 0.0F, 1.0F);
        lastObservation[23] = (!p.canSee(t) ? 1.0F : 0.0F) / 6.0F;
        lastObservation[24] = clip(countInventoryItem(p, Items.END_CRYSTAL) / 6.0F, 0.0F, 1.0F);
        lastObservation[25] = clip(countInventoryItem(p, Items.RESPAWN_ANCHOR) / 4.0F, 0.0F, 1.0F);
        lastObservation[26] = clip(countInventoryItem(p, Items.TNT_MINECART) / 4.0F, 0.0F, 1.0F);
        int enemyExplosives = opponent == null ? 0
                : countInventoryItem(opponent, Items.END_CRYSTAL)
                + countInventoryItem(opponent, Items.RESPAWN_ANCHOR)
                + countInventoryItem(opponent, Items.TNT_MINECART);
        lastObservation[27] = clip(enemyExplosives / 10.0F, 0.0F, 1.0F);
        lastObservation[28] = clip(countInventoryItem(p, Items.OBSIDIAN) / 16.0F, 0.0F, 1.0F);
        lastObservation[29] = hasObsidianCage(p, mc) ? 1.0F : 0.0F;
        lastObservation[30] = clip(playerComboTaken / 5.0F, 0.0F, 1.0F);
        float stun = playerComboTaken == 0 ? 0.0F
                : Math.max(0.0F, 0.45F - Math.max(0, tick - lastPlayerDamageTick) * 0.05F);
        lastObservation[31] = clip(stun, 0.0F, 1.0F);
        lastObservation[32] = clip(enemyComboTaken / 5.0F, 0.0F, 1.0F);
        lastObservation[33] = playerComboTaken == 0 ? 0.0F
                : clip(p.hurtTime * 0.05F, 0.0F, 1.0F);
    }

    private static void updateComboState(ClientPlayerEntity p, LivingEntity t) {
        int tick = p.tickCount;
        float health = p.getHealth() + p.getAbsorptionAmount();
        float targetHealth = t.getHealth() + t.getAbsorptionAmount();
        if (!Float.isNaN(previousHealth) && health < previousHealth - 0.01F) {
            playerComboTaken = tick - lastPlayerDamageTick <= 32 ? playerComboTaken + 1 : 1;
            lastPlayerDamageTick = tick;
        }
        if (!Float.isNaN(previousTargetHealth) && targetHealth < previousTargetHealth - 0.01F
                && horizontalDistance(p, t) <= 4.5) {
            enemyComboTaken = tick - lastEnemyDamageTick <= 32 ? enemyComboTaken + 1 : 1;
            lastEnemyDamageTick = tick;
            enemyStunUntilTick = tick + 9;
        }
        previousHealth = health;
        previousTargetHealth = targetHealth;
        if (tick - lastPlayerDamageTick > 32) playerComboTaken = 0;
        if (tick - lastEnemyDamageTick > 32) enemyComboTaken = 0;
        if (tick > counterUntilTick) counterUntilTick = 0;
        if (tick > counterArmedUntilTick) counterArmedUntilTick = 0;
    }

    private static ItemStack observationWeapon(LivingEntity entity, int preferredSlot) {
        if (preferredSlot >= 0 && entity instanceof PlayerEntity) {
            PlayerEntity player = (PlayerEntity) entity;
            if (preferredSlot < 9 && preferredSlot < player.inventory.items.size()) {
                ItemStack stack = player.inventory.items.get(preferredSlot);
                if (!stack.isEmpty()) return stack;
            }
        }
        return entity.getMainHandItem();
    }

    private static float weaponDamage(LivingEntity entity, int preferredSlot) {
        ItemStack held = observationWeapon(entity, preferredSlot);
        double value = 1.0;
        if (!held.isEmpty()) {
            for (AttributeModifier modifier : held.getAttributeModifiers(EquipmentSlotType.MAINHAND)
                    .get(Attributes.ATTACK_DAMAGE)) {
                value += modifier.getAmount();
            }
        }
        if (!(entity instanceof PlayerEntity) || preferredSlot < 0) {
            try {
                double attribute = entity.getAttributeValue(Attributes.ATTACK_DAMAGE);
                if (Double.isFinite(attribute) && attribute >= 0.0) value = attribute;
            } catch (Exception ignored) { }
        }
        return (float) Math.max(0.0, Math.min(14.0, value));
    }

    private static float weaponCooldown(LivingEntity entity, int preferredSlot) {
        ItemStack held = observationWeapon(entity, preferredSlot);
        if (entity instanceof PlayerEntity && preferredSlot >= 0) {
            double speed = 4.0;
            if (!held.isEmpty()) {
                for (AttributeModifier modifier : held.getAttributeModifiers(EquipmentSlotType.MAINHAND)
                        .get(Attributes.ATTACK_SPEED)) {
                    speed += modifier.getAmount();
                }
            }
            if (speed > 0.0) return (float) (1.0 / speed);
        }
        try {
            double speed = entity.getAttributeValue(Attributes.ATTACK_SPEED);
            if (Double.isFinite(speed) && speed > 0.0) return (float) (1.0 / speed);
        } catch (Exception ignored) { }
        return 0.6F;
    }

    private static float armorTierReduction(int armor) {
        if (armor <= 0) return 0.0F;
        if (armor <= 7) return 0.16F;
        if (armor <= 12) return 0.36F;
        if (armor <= 16) return 0.56F;
        return 0.72F;
    }

    private static boolean isShieldBlocking(LivingEntity entity) {
        return entity.isBlocking()
                && (entity.getMainHandItem().getItem() instanceof ShieldItem
                    || entity.getOffhandItem().getItem() instanceof ShieldItem);
    }

    private static int countInventoryItem(PlayerEntity player, Item item) {
        if (player == null || item == null) return 0;
        int count = 0;
        for (ItemStack stack : player.inventory.items) {
            if (!stack.isEmpty() && stack.getItem() == item) count += stack.getCount();
        }
        for (ItemStack stack : player.inventory.offhand) {
            if (!stack.isEmpty() && stack.getItem() == item) count += stack.getCount();
        }
        return count;
    }

    private static int countHealingApples(PlayerEntity player) {
        return countInventoryItem(player, Items.GOLDEN_APPLE)
                + countInventoryItem(player, Items.ENCHANTED_GOLDEN_APPLE);
    }

    private static int countSplashPotions(PlayerEntity player, Effect effect) {
        if (player == null) return 0;
        int count = 0;
        for (ItemStack stack : player.inventory.items) {
            if (stack.getItem() instanceof SplashPotionItem && hasPotionEffect(stack, effect)) {
                count += stack.getCount();
            }
        }
        return count;
    }

    private static int countPotionsWithEffect(PlayerEntity player, Effect effect) {
        if (player == null) return 0;
        int count = 0;
        for (ItemStack stack : player.inventory.items) {
            if (stack.getItem() instanceof net.minecraft.item.PotionItem && hasPotionEffect(stack, effect)) {
                count += stack.getCount();
            }
        }
        return count;
    }

    private static boolean hasPotionEffect(ItemStack stack, Effect effect) {
        if (stack.isEmpty()) return false;
        for (EffectInstance instance : PotionUtils.getPotion(stack).getEffects()) {
            if (instance.getEffect() == effect) return true;
        }
        return false;
    }

    private static int countBuildingBlocks(PlayerEntity player) {
        if (player == null) return 0;
        int count = 0;
        for (ItemStack stack : player.inventory.items) {
            if (!stack.isEmpty() && stack.getItem() instanceof BlockItem
                    && stack.getItem() != Items.OBSIDIAN
                    && stack.getItem() != Items.COBWEB
                    && stack.getItem() != Items.RESPAWN_ANCHOR) {
                count += stack.getCount();
            }
        }
        return count;
    }

    private static boolean hasObsidianCage(ClientPlayerEntity p, Minecraft mc) {
        BlockPos center = p.blockPosition();
        int count = 0;
        for (int x = -1; x <= 1; x++) {
            for (int z = -1; z <= 1; z++) {
                if (Math.abs(x) + Math.abs(z) != 1) continue;
                for (int y = 0; y <= 1; y++) {
                    if (mc.level.getBlockState(center.offset(x, y, z)).getBlock() == Blocks.OBSIDIAN) count++;
                }
            }
        }
        return count >= 4;
    }

    private static double horizontalDistance(LivingEntity a, LivingEntity b) {
        double dx = a.getX() - b.getX();
        double dz = a.getZ() - b.getZ();
        return Math.sqrt(dx * dx + dz * dz);
    }

    private static float clip(float value, float low, float high) {
        if (Float.isNaN(value) || Float.isInfinite(value)) return low;
        return Math.max(low, Math.min(high, value));
    }

    private static void applyPolicyAction(Minecraft mc, ClientPlayerEntity p, LivingEntity t, int action) {
        if (shieldActionActive && action != 12) {
            p.stopUsingItem();
            shieldActionActive = false;
        }
        setMovementForAction(action);

        // Eating/drinking is a timed vanilla action. Keep it intact while still
        // allowing the PPO actor to control movement; item/attack actions wait.
        if (isLongItemUse(p)) {
            p.setSprinting(inSprint);
            return;
        }
        if (actionCooldownUntilTick[action] > p.tickCount) {
            p.setSprinting(inSprint);
            return;
        }
        boolean hadResource = hasHotbarResource(p, action);

        switch (action) {
            case 0: break;
            case 1: case 2: case 3: case 4: case 8: case 10: break;
            case 5: case 6: case 7: case 9: case 11:
                attackTarget(mc, p, t);
                break;
            case 12:
                raisePpoShield(mc, p);
                break;
            case 13:
                useHotbarItem(mc, p, Items.GOLDEN_APPLE, Items.ENCHANTED_GOLDEN_APPLE);
                break;
            case 14:
                usePotion(mc, p, Effects.HEAL, true);
                break;
            case 15:
                useHotbarItem(mc, p, Items.ENDER_PEARL);
                break;
            case 16:
                if (!usePotion(mc, p, Effects.DAMAGE_BOOST, false)) {
                    usePotion(mc, p, Effects.MOVEMENT_SPEED, false);
                }
                break;
            case 17:
                useRangedItem(mc, p);
                break;
            case 18:
                placeBarricade(mc, p, t);
                break;
            case 19:
                placeTrap(mc, p, t);
                break;
            case 20:
                useWaterBucket(mc, p);
                break;
            case 21:
                placePillarBlock(mc, p);
                break;
            case 22:
                useCrystalCombo(mc, p, t, false);
                break;
            case 23:
                useRespawnAnchor(mc, p, t);
                break;
            case 24:
                useTntMinecart(mc, p, t);
                break;
            case 25:
                placeObsidianCage(mc, p);
                break;
            case 26:
                attackTarget(mc, p, t);
                break;
            case 27:
                useCrystalCombo(mc, p, t, true);
                break;
            case 28:
                breakCombo(mc, p);
                break;
            case 29:
                counterCombo(mc, p, t);
                break;
            default:
                setMovementForAction(0);
                break;
        }
        int cooldown = actionCooldownTicks(action);
        if (hadResource && cooldown > 0) actionCooldownUntilTick[action] = p.tickCount + cooldown;
        p.setSprinting(inSprint);
    }

    private static int actionCooldownTicks(int action) {
        switch (action) {
            case 13: return 7;
            case 14: return 4;
            case 15: case 27: return 30;
            case 16: case 18: case 19: case 20: return 5;
            case 17: return 3;
            case 21: return 4;
            case 22: case 23: case 25: return 6;
            case 24: return 7;
            default: return 0;
        }
    }

    private static boolean hasHotbarResource(ClientPlayerEntity p, int action) {
        switch (action) {
            case 13:
                return findHotbarSlot(p, Items.GOLDEN_APPLE) >= 0
                        || findHotbarSlot(p, Items.ENCHANTED_GOLDEN_APPLE) >= 0;
            case 14: return findHotbarPotion(p, Effects.HEAL, true) >= 0;
            case 15: return findHotbarSlot(p, Items.ENDER_PEARL) >= 0;
            case 16: return findHotbarPotion(p, Effects.DAMAGE_BOOST, false) >= 0
                        || findHotbarPotion(p, Effects.MOVEMENT_SPEED, false) >= 0;
            case 17:
                return findHotbarPotion(p, Effects.HARM, true) >= 0
                        || findHotbarSlot(p, Items.SNOWBALL) >= 0 || findHotbarSlot(p, Items.EGG) >= 0
                        || findHotbarSlot(p, Items.CROSSBOW) >= 0 || findHotbarSlot(p, Items.BOW) >= 0;
            case 18: case 21: return findHotbarBlock(p, false) >= 0;
            case 19: return findHotbarSlot(p, Items.COBWEB) >= 0
                        || findHotbarSlot(p, Items.LAVA_BUCKET) >= 0;
            case 20: return findHotbarSlot(p, Items.WATER_BUCKET) >= 0;
            case 22: return findHotbarSlot(p, Items.END_CRYSTAL) >= 0
                        && findHotbarSlot(p, Items.OBSIDIAN) >= 0;
            case 23: return findHotbarSlot(p, Items.RESPAWN_ANCHOR) >= 0;
            case 24: return findHotbarSlot(p, Items.RAIL) >= 0
                        && findHotbarSlot(p, Items.TNT_MINECART) >= 0;
            case 25: return countHotbarItem(p, Items.OBSIDIAN) >= 4;
            case 27: return findHotbarSlot(p, Items.ENDER_PEARL) >= 0
                        && findHotbarSlot(p, Items.END_CRYSTAL) >= 0
                        && findHotbarSlot(p, Items.OBSIDIAN) >= 0;
            default: return false;
        }
    }

    private static int countHotbarItem(PlayerEntity p, Item item) {
        int count = 0;
        for (int i = 0; i < 9; i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (!stack.isEmpty() && stack.getItem() == item) count += stack.getCount();
        }
        return count;
    }

    private static void setMovementForAction(int action) {
        inForward = 0.0F;
        inStrafe = 0.0F;
        inJump = false;
        inSprint = false;
        switch (action) {
            case 1: inForward = 1.0F; break;
            case 2: inForward = -1.0F; break;
            case 3: inStrafe = 1.0F; break;
            case 4: inStrafe = -1.0F; break;
            case 5: inForward = 1.0F; break;
            case 6: inStrafe = 1.0F; break;
            case 7: inStrafe = -1.0F; break;
            case 8: inJump = true; break;
            case 9: inForward = 1.0F; inJump = true; inSprint = true; break;
            case 10: inForward = 1.0F; inSprint = true; break;
            case 11: inForward = -1.0F; break;
            case 12: inForward = -0.45F; break;
            case 18: case 22: case 28: inForward = -0.45F; break;
            case 21: inJump = true; break;
            case 29: inForward = 0.4F; break;
            default: break;
        }
    }

    private static boolean isLongItemUse(ClientPlayerEntity p) {
        if (!p.isUsingItem()) return false;
        ItemStack used = p.getUseItem();
        return !used.isEmpty() && (used.getItem().isEdible()
                || used.getItem() instanceof net.minecraft.item.PotionItem
                || used.getItem() instanceof net.minecraft.item.BowItem
                || used.getItem() instanceof net.minecraft.item.CrossbowItem);
    }

    private static void releaseChargedRangedItem(Minecraft mc, ClientPlayerEntity p) {
        if (rangedReleaseTick > 0 && p.tickCount >= rangedReleaseTick && p.isUsingItem()) {
            mc.gameMode.releaseUsingItem(p);
            rangedReleaseTick = 0;
        } else if (rangedReleaseTick > 0 && !p.isUsingItem()) {
            rangedReleaseTick = 0;
        }
    }

    private static void attackTarget(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        if (t == null || !t.isAlive() || horizontalDistance(p, t) > 3.45
                || !p.canSee(t) || p.isUsingItem()
                || p.getAttackStrengthScale(0.5F) < 0.95F) return;
        if (combatWeaponSlot >= 0 && p.inventory.selected != combatWeaponSlot) {
            p.inventory.selected = combatWeaponSlot;
        }
        mc.gameMode.attack(p, t);
        p.swing(Hand.MAIN_HAND);
        lastMeleeAttackTick = p.tickCount;
        enemyStunUntilTick = p.tickCount + 9;
        if (lastAction == 26) p.setSprinting(false); // vanilla sprint reset; the simulator's extra bonus is not forge-exposed
    }

    private static void raisePpoShield(Minecraft mc, ClientPlayerEntity p) {
        if (p.isBlocking()) {
            shieldActionActive = true;
            return;
        }
        Hand hand = null;
        if (p.getOffhandItem().getItem() instanceof ShieldItem) {
            hand = Hand.OFF_HAND;
        } else if (p.getMainHandItem().getItem() instanceof ShieldItem) {
            hand = Hand.MAIN_HAND;
        } else {
            int slot = findHotbarSlot(p, Items.SHIELD);
            if (slot >= 0) {
                select(p, slot);
                hand = Hand.MAIN_HAND;
            }
        }
        if (hand != null) {
            mc.gameMode.useItem(p, p.level, hand);
            shieldActionActive = true;
        }
    }

    private static boolean useHotbarItem(Minecraft mc, ClientPlayerEntity p, Item... items) {
        for (Item item : items) {
            int slot = findHotbarSlot(p, item);
            if (slot < 0) continue;
            select(p, slot);
            mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
            return true;
        }
        return false;
    }

    private static boolean usePotion(Minecraft mc, ClientPlayerEntity p, Effect effect, boolean splashOnly) {
        int slot = findHotbarPotion(p, effect, splashOnly);
        if (slot < 0) return false;
        select(p, slot);
        mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
        return true;
    }

    private static int findHotbarPotion(ClientPlayerEntity p, Effect effect, boolean splashOnly) {
        for (int i = 0; i < 9; i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (!(stack.getItem() instanceof net.minecraft.item.PotionItem)) continue;
            if (splashOnly && !(stack.getItem() instanceof SplashPotionItem)) continue;
            if (!splashOnly && stack.getItem() instanceof SplashPotionItem) continue;
            if (hasPotionEffect(stack, effect)) return i;
        }
        return -1;
    }

    private static void useRangedItem(Minecraft mc, ClientPlayerEntity p) {
        int slot = findHotbarPotion(p, Effects.HARM, true);
        if (slot >= 0) {
            select(p, slot);
            mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
            return;
        }
        if (useHotbarItem(mc, p, Items.SNOWBALL, Items.EGG)) return;
        if (useHotbarItem(mc, p, Items.CROSSBOW)) {
            rangedReleaseTick = p.tickCount + 25;
        } else if (useHotbarItem(mc, p, Items.BOW)) {
            rangedReleaseTick = p.tickCount + 20;
        }
    }

    private static int findHotbarSlot(PlayerEntity p, Item item) {
        for (int i = 0; i < 9; i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (!stack.isEmpty() && stack.getItem() == item) return i;
        }
        return -1;
    }

    private static int findHotbarBlock(PlayerEntity p, boolean obsidianOnly) {
        for (int i = 0; i < 9; i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (stack.isEmpty() || !(stack.getItem() instanceof BlockItem)) continue;
            if (obsidianOnly && stack.getItem() != Items.OBSIDIAN) continue;
            if (!obsidianOnly && (stack.getItem() == Items.OBSIDIAN || stack.getItem() == Items.COBWEB
                    || stack.getItem() == Items.RESPAWN_ANCHOR)) continue;
            return i;
        }
        return -1;
    }

    private static boolean placeBlockAt(Minecraft mc, ClientPlayerEntity p, BlockPos desired, Item item) {
        int slot = findHotbarSlot(p, item);
        if (slot < 0 || p.isUsingItem()) return false;
        select(p, slot);
        Direction[] faces = {Direction.DOWN, Direction.NORTH, Direction.SOUTH,
                Direction.WEST, Direction.EAST, Direction.UP};
        for (Direction towardSupport : faces) {
            BlockPos support = desired.relative(towardSupport);
            if (mc.level.getBlockState(support).isAir()) continue;
            Direction face = towardSupport.getOpposite();
            Vector3d hit = new Vector3d(support.getX() + 0.5 + face.getStepX() * 0.5,
                    support.getY() + 0.5 + face.getStepY() * 0.5,
                    support.getZ() + 0.5 + face.getStepZ() * 0.5);
            mc.gameMode.useItemOn(p, mc.level, Hand.MAIN_HAND,
                    new BlockRayTraceResult(hit, face, support, false));
            return true;
        }
        return false;
    }

    private static void useOnBlock(Minecraft mc, ClientPlayerEntity p, BlockPos block, Direction face) {
        Vector3d hit = new Vector3d(block.getX() + 0.5 + face.getStepX() * 0.5,
                block.getY() + 0.5 + face.getStepY() * 0.5,
                block.getZ() + 0.5 + face.getStepZ() * 0.5);
        mc.gameMode.useItemOn(p, mc.level, Hand.MAIN_HAND,
                new BlockRayTraceResult(hit, face, block, false));
    }

    private static void useBucketAt(Minecraft mc, ClientPlayerEntity p, Item bucket, BlockPos desired) {
        int slot = findHotbarSlot(p, bucket);
        if (slot < 0 || p.isUsingItem()) return;
        select(p, slot);
        BlockPos support = desired.below();
        if (!mc.level.getBlockState(support).isAir()) useOnBlock(mc, p, support, Direction.UP);
    }

    private static void placeBarricade(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        int slot = findHotbarBlock(p, false);
        if (slot < 0 || p.distanceTo(t) < 1.6 || p.distanceTo(t) > 7.5) return;
        Item block = p.inventory.items.get(slot).getItem();
        BlockPos mid = new BlockPos((p.getX() + t.getX()) * 0.5, t.getY(), (p.getZ() + t.getZ()) * 0.5);
        placeBlockAt(mc, p, mid, block);
    }

    private static void placeTrap(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        double dist = horizontalDistance(p, t);
        if (dist <= 4.5) {
            int web = findHotbarSlot(p, Items.COBWEB);
            if (web >= 0) {
                BlockPos desired = t.blockPosition();
                placeBlockAt(mc, p, desired, Items.COBWEB);
                return;
            }
        }
        if (dist >= 1.9 && dist <= 4.2) {
            useBucketAt(mc, p, Items.LAVA_BUCKET, t.blockPosition());
        }
    }

    private static void useWaterBucket(Minecraft mc, ClientPlayerEntity p) {
        useBucketAt(mc, p, Items.WATER_BUCKET, p.blockPosition());
    }

    private static void placePillarBlock(Minecraft mc, ClientPlayerEntity p) {
        int slot = findHotbarBlock(p, false);
        if (slot < 0) return;
        Item block = p.inventory.items.get(slot).getItem();
        BlockPos desired = p.blockPosition().relative(Direction.fromYRot(p.yRot).getOpposite());
        placeBlockAt(mc, p, desired, block);
    }

    private static BlockPos targetAdjacentBlock(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        BlockPos center = t.blockPosition();
        Direction towardPlayer = Direction.fromYRot(p.yRot).getOpposite();
        Direction[] candidates = {towardPlayer, Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST};
        for (Direction direction : candidates) {
            BlockPos candidate = center.relative(direction);
            if (mc.level.getBlockState(candidate).isAir()) return candidate;
        }
        return center;
    }

    private static void useCrystalCombo(Minecraft mc, ClientPlayerEntity p, LivingEntity t, boolean pearlFirst) {
        double dist = horizontalDistance(p, t);
        if (pearlFirst) {
            if (dist < 3.6 || dist > 12.0 || findHotbarSlot(p, Items.ENDER_PEARL) < 0
                    || findHotbarSlot(p, Items.END_CRYSTAL) < 0
                    || findHotbarSlot(p, Items.OBSIDIAN) < 0) return;
            useHotbarItem(mc, p, Items.ENDER_PEARL);
        } else if (dist < 2.4 || dist > 5.5
                || findHotbarSlot(p, Items.END_CRYSTAL) < 0
                || findHotbarSlot(p, Items.OBSIDIAN) < 0) {
            return;
        }
        BlockPos base = targetAdjacentBlock(mc, p, t);
        if (!placeBlockAt(mc, p, base, Items.OBSIDIAN)) return;
        int crystalSlot = findHotbarSlot(p, Items.END_CRYSTAL);
        if (crystalSlot < 0) return;
        select(p, crystalSlot);
        useOnBlock(mc, p, base, Direction.UP);
        placedCrystalBase = base;
        crystalDetonateUntilTick = p.tickCount + 30;
    }

    private static void detonatePlacedCrystal(Minecraft mc, ClientPlayerEntity p) {
        if (placedCrystalBase == null || p.tickCount > crystalDetonateUntilTick) return;
        AxisAlignedBB box = new AxisAlignedBB(placedCrystalBase.getX() - 2.0, placedCrystalBase.getY() - 1.0,
                placedCrystalBase.getZ() - 2.0, placedCrystalBase.getX() + 3.0,
                placedCrystalBase.getY() + 4.0, placedCrystalBase.getZ() + 3.0);
        List<EnderCrystalEntity> crystals = mc.level.getEntitiesOfClass(EnderCrystalEntity.class, box,
                e -> e.isAlive());
        if (!crystals.isEmpty() && p.getAttackStrengthScale(0.5F) >= 0.95F) {
            mc.gameMode.attack(p, crystals.get(0));
            p.swing(Hand.MAIN_HAND);
            lastMeleeAttackTick = p.tickCount;
            placedCrystalBase = null;
        }
    }

    private static void useRespawnAnchor(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        double dist = horizontalDistance(p, t);
        if (dist < 1.5 || dist > 4.8 || findHotbarSlot(p, Items.RESPAWN_ANCHOR) < 0) return;
        BlockPos anchor = targetAdjacentBlock(mc, p, t);
        if (!placeBlockAt(mc, p, anchor, Items.RESPAWN_ANCHOR)) return;
        int glowstone = findHotbarSlot(p, Items.GLOWSTONE);
        if (glowstone >= 0) {
            select(p, glowstone);
            useOnBlock(mc, p, anchor, Direction.UP);
            if (combatWeaponSlot >= 0) select(p, combatWeaponSlot);
            useOnBlock(mc, p, anchor, Direction.UP);
        }
    }

    private static void useTntMinecart(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        double dist = horizontalDistance(p, t);
        if (dist < 3.0 || dist > 8.5 || findHotbarSlot(p, Items.RAIL) < 0
                || findHotbarSlot(p, Items.TNT_MINECART) < 0) return;
        BlockPos rail = targetAdjacentBlock(mc, p, t);
        if (!placeBlockAt(mc, p, rail, Items.RAIL)) return;
        int cart = findHotbarSlot(p, Items.TNT_MINECART);
        if (cart >= 0) {
            select(p, cart);
            useOnBlock(mc, p, rail, Direction.UP);
        }
    }

    private static void placeObsidianCage(Minecraft mc, ClientPlayerEntity p) {
        if (countInventoryItem(p, Items.OBSIDIAN) < 4) return;
        BlockPos center = p.blockPosition();
        BlockPos[] walls = {center.north(), center.south(), center.east(), center.west()};
        for (BlockPos wall : walls) placeBlockAt(mc, p, wall, Items.OBSIDIAN);
    }

    private static void breakCombo(Minecraft mc, ClientPlayerEntity p) {
        if (playerComboTaken <= 0 || p.tickCount - lastPlayerDamageTick > 32) return;
        if (p.getOffhandItem().getItem() instanceof ShieldItem
                || p.getMainHandItem().getItem() instanceof ShieldItem
                || findHotbarSlot(p, Items.SHIELD) >= 0) {
            raisePpoShield(mc, p);
        } else if (findHotbarSlot(p, Items.WATER_BUCKET) >= 0) {
            useWaterBucket(mc, p);
        } else if (findHotbarSlot(p, Items.ENDER_PEARL) >= 0) {
            useHotbarItem(mc, p, Items.ENDER_PEARL);
        }
        playerComboTaken = 0;
        lastPlayerDamageTick = p.tickCount;
        counterUntilTick = p.tickCount + 36;
        counterArmedUntilTick = 0;
        inForward = -1.0F;
    }

    private static void counterCombo(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        boolean punishable = p.tickCount <= counterUntilTick || p.tickCount <= enemyStunUntilTick
                || p.tickCount <= counterArmedUntilTick;
        if (!punishable) return;
        counterArmedUntilTick = p.tickCount + 24;
        if (horizontalDistance(p, t) <= 4.0 && p.getAttackStrengthScale(0.5F) >= 0.95F) {
            counterArmedUntilTick = 0;
            attackTarget(mc, p, t);
            inForward = 0.4F;
        }
    }

    // -------------------------------------------------------------- targeting
    private static LivingEntity findTarget(Minecraft mc, ClientPlayerEntity p) {
        AxisAlignedBB box = p.getBoundingBox().inflate(searchRange, 6.0, searchRange);
        List<LivingEntity> list = mc.level.getEntitiesOfClass(LivingEntity.class, box,
                e -> e != p && e.isAlive() && e.isPickable() && !(e instanceof ArmorStandEntity)
                        && !(e instanceof PlayerEntity && (((PlayerEntity) e).isCreative()
                             || ((PlayerEntity) e).isSpectator())));
        LivingEntity best = null;
        double bestScore = Double.MAX_VALUE;
        Vector3d eye = p.position();
        for (LivingEntity e : list) {
            double d = p.distanceTo(e);
            if (d > searchRange) continue;
            // приоритет: игроки и мобы в поле зрения, ближайшие
            double score = d;
            if (!p.canSee(e)) score += 4.0;
            if (e instanceof PlayerEntity) score -= 1.5;
            if (score < bestScore) { bestScore = score; best = e; }
            if (eye.distanceToSqr(e.position()) < 0.0001) continue;
        }
        return best;
    }

    // ------------------------------------------------------------------- aim
    private static void aimAt(ClientPlayerEntity p, LivingEntity t, double dist) {
        double dx = t.getX() - p.getX();
        double dy = (t.getY() + t.getBbHeight() * 0.85) - (p.getY() + p.getEyeHeight());
        double dz = t.getZ() - p.getZ();
        double horiz = Math.sqrt(dx * dx + dz * dz);
        float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx))) - 90.0F;
        float pitch = (float) (-Math.toDegrees(Math.atan2(dy, horiz)));
        if (dist > 6.0) pitch *= 0.6F;
        p.yRot = yaw;
        p.xRot = pitch;
        p.yHeadRot = yaw;
        p.yBodyRot = yaw;
    }

    // -------------------------------------------------------------- weapons
    private static void selectBestWeapon(ClientPlayerEntity p) {
        if (p.isUsingItem()) return;
        int best = -1;
        double bestDmg = -1.0;
        for (int i = 0; i < 9; i++) {
            ItemStack s = p.inventory.items.get(i);
            if (s.isEmpty()) continue;
            Item it = s.getItem();
            if (!(it instanceof SwordItem || it instanceof AxeItem || it instanceof TridentItem)) continue;
            double dmg = attackDamage(s);
            if (dmg > bestDmg) { bestDmg = dmg; best = i; }
        }
        if (best >= 0 && best != p.inventory.selected) select(p, best);
    }

    private static double attackDamage(ItemStack stack) {
        double sum = 0.0;
        for (AttributeModifier m : stack.getAttributeModifiers(EquipmentSlotType.MAINHAND)
                .get(Attributes.ATTACK_DAMAGE)) {
            sum += m.getAmount();
        }
        return sum;
    }

    private static void select(ClientPlayerEntity p, int slot) {
        p.inventory.selected = slot;   // клиент синхронизирует слот сам
    }

    private static void releaseKeys(Minecraft mc) {
        ClientPlayerEntity p = mc.player;
        if (shieldActionActive && p != null) p.stopUsingItem();
        shieldActionActive = false;
        inForward = 0.0F;
        inStrafe = 0.0F;
        inJump = false;
        inSprint = false;
        if (p != null) p.setSprinting(false);
    }

    /**
     * Перехват управления: пока автобой включён, ванильные клавиши игнорируются,
     * а в игру уходят только интенты «мозга». Клиент штатно отправит их на сервер.
     */
    public static void overrideInputs(MovementInput in) {
        if (!AiPilot.isEnabled() || in == null) return;
        Minecraft mc = Minecraft.getInstance();
        if (mc.player == null || mc.screen != null) return;
        in.forwardImpulse = inForward;
        in.leftImpulse = inStrafe;
        in.up = inForward > 0.05F || inJump;
        in.down = inForward < -0.05F;
        in.left = inStrafe > 0.05F;
        in.right = inStrafe < -0.05F;
        in.jumping = inJump;
        mc.player.setSprinting(inSprint);
    }

    // ------------------------------------------------------------------ HUD
    public static void renderHud(RenderGameOverlayEvent.Text event) {
        Minecraft mc = Minecraft.getInstance();
        ClientPlayerEntity p = mc.player;
        if (p == null) return;
        List<String> lines = event.getLeft();
        String model = ppoPolicy.isReady()
                ? "\u00A7bPPO action " + (lastAction < 0 ? "-" : lastAction)
                : "\u00A7cPPO WEIGHTS MISSING";
        lines.add("\u00A7bAI Pilot " + (AiPilot.isEnabled() ? "\u00A7aON" : "\u00A7cOFF")
                + "\u00A77  | " + model);

        if (AiPilot.isEnabled() && target != null) {
            double d = p.distanceTo(target);
            String hp = String.format(Locale.ROOT, "%.1f", target.getHealth());
            lines.add("\u00A7e  target: " + target.getName().getString() + "  " + hp + " HP  "
                    + String.format(Locale.ROOT, "%.1f", d) + "m");
            int charge = Math.round(p.getAttackStrengthScale(0.5F) * 100.0F);
            lines.add((charge > 95 ? "\u00A7a" : "\u00A77") + "  charge: " + charge + "%");
            if (playerComboTaken > 0) {
                lines.add("\u00A7d  incoming combo: " + playerComboTaken + "  | counter window: "
                        + (p.tickCount <= counterUntilTick ? "open" : "closed"));
            }
        } else if (AiPilot.isEnabled()) {
            lines.add("\u00A78  no target in " + (int) searchRange + "m");
        }
    }
}

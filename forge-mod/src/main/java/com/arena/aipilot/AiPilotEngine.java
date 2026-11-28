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
import net.minecraft.inventory.container.ClickType;
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
    private static final float[] lastObservation = new float[52];
    private static final int[] actionCooldownUntilTick = new int[30];
    private static final double[] terrainSampleDistances = {0.75, 1.5, 2.25, 3.0};
    private static TerrainScan cachedTerrainScan;
    private static int cachedTerrainTick = Integer.MIN_VALUE;
    private static int cachedTerrainTargetId = -1;
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
    private static int pendingCritTargetId = -1;
    private static int pendingCritUntilTick = -1;
    private static int lastCritAttackTick = -1_000_000;
    private static int forcedAimUntilTick = -1;
    private static float forcedAimYaw = 0.0F;
    private static float forcedAimPitch = 0.0F;
    private static int forcedRetreatUntilTick = -1;
    private static int forcedRetreatTargetId = -1;
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
        if ((shieldActionActive || rangedReleaseTick > 0) && localPlayer != null) {
            localPlayer.stopUsingItem();
        }
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
        pendingCritTargetId = -1;
        pendingCritUntilTick = -1;
        lastCritAttackTick = -1_000_000;
        forcedAimUntilTick = -1;
        forcedAimYaw = 0.0F;
        forcedAimPitch = 0.0F;
        enemyStunUntilTick = 0;
        crystalDetonateUntilTick = 0;
        rangedReleaseTick = 0;
        placedCrystalBase = null;
        shieldActionActive = false;
        lastPolicyTick = Integer.MIN_VALUE;
        cachedTerrainScan = null;
        cachedTerrainTick = Integer.MIN_VALUE;
        cachedTerrainTargetId = -1;
        forcedRetreatUntilTick = -1;
        forcedRetreatTargetId = -1;
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
        if (rangedReleaseTick > 0) aimAtProjectileTarget(p, target);
        else aimAtTargetOrForced(p, target, dist);
        updateComboState(p, target);
        detonatePlacedCrystal(mc, p);

        // Training DT is 0.10s. Forge ticks are 0.05s, so infer once every two ticks.
        if ((p.tickCount & 1) == 0 && lastPolicyTick != p.tickCount) {
            lastPolicyTick = p.tickCount;
            buildObservation(p, target, mc);
            int proposedAction = ppoPolicy.predict(lastObservation);
            lastAction = selectTacticalAction(p, target, proposedAction);
            modelStepCount = Math.min(600, modelStepCount + 1);
            applyPolicyAction(mc, p, target, lastAction);
        }
        processPendingCrit(mc, p, target);
        applyNavigationAssist(mc, p, target);
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
        pendingCritTargetId = -1;
        pendingCritUntilTick = -1;
        lastCritAttackTick = -1_000_000;
        forcedAimUntilTick = -1;
        forcedAimYaw = 0.0F;
        forcedAimPitch = 0.0F;
        enemyStunUntilTick = 0;
        crystalDetonateUntilTick = 0;
        placedCrystalBase = null;
        lastAction = -1;
        lastPolicyTick = Integer.MIN_VALUE;
        cachedTerrainScan = null;
        cachedTerrainTick = Integer.MIN_VALUE;
        cachedTerrainTargetId = -1;
        forcedRetreatUntilTick = -1;
        forcedRetreatTargetId = -1;
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
                + countInventoryItem(p, Items.SPECTRAL_ARROW) + countInventoryItem(p, Items.BOW)
                + countInventoryItem(p, Items.CROSSBOW) + countSplashPotions(p, Effects.HARM);
        float rangedFeature = clip((buffs + ranged) / 12.0F, 0.0F, 0.60F);
        boolean hasBowAmmo = hasArrowAmmo(p) && countInventoryItem(p, Items.BOW) > 0;
        boolean hasCrossbowAmmo = hasArrowAmmo(p) && countInventoryItem(p, Items.CROSSBOW) > 0;
        if (hasBowAmmo) rangedFeature = 0.75F;
        if (hasCrossbowAmmo) rangedFeature = 0.85F;
        if (findCrossbowInventorySlot(p, true) >= 0) rangedFeature = 1.10F;
        if (rangedReleaseTick > 0 && p.isUsingItem()) rangedFeature = 1.25F;
        lastObservation[18] = rangedFeature;
        lastObservation[19] = isShieldBlocking(p) ? 1.0F : 0.0F;
        lastObservation[20] = isShieldBlocking(t) ? 1.0F : 0.0F;
        lastObservation[21] = clip(countBuildingBlocks(p) / 24.0F, 0.0F, 1.0F);
        int utilities = countInventoryItem(p, Items.COBWEB) + countInventoryItem(p, Items.LAVA_BUCKET)
                + countInventoryItem(p, Items.WATER_BUCKET);
        lastObservation[22] = clip(utilities / 6.0F, 0.0F, 1.0F);
        lastObservation[23] = (!p.canSee(t) ? 1.0F : 0.0F) / 6.0F;
        lastObservation[24] = clip(countInventoryItem(p, Items.END_CRYSTAL) / 6.0F, 0.0F, 1.0F);
        // Preserve the actor's inventory-count features; action handlers separately
        // guard the dimension and full ingredient requirements before using explosives.
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
        TerrainScan terrain = getTerrainScan(mc, p, t);
        for (int i = 0; i < 4; i++) {
            lastObservation[34 + i] = terrain.clearance[i];
            lastObservation[38 + i] = terrain.support[i];
        }
        lastObservation[42] = terrain.targetHeightDelta;
        lastObservation[43] = terrain.forwardSlope;
        populateCrowdObservation(p, t, mc);
    }

    /** Append the four fixed crowd features used by the one-versus-many PPO actor. */
    private static void populateCrowdObservation(ClientPlayerEntity p, LivingEntity target, Minecraft mc) {
        AxisAlignedBB box = p.getBoundingBox().inflate(searchRange, 6.0, searchRange);
        List<LivingEntity> nearby = new java.util.ArrayList<>(
                mc.level.getEntitiesOfClass(LivingEntity.class, box,
                        e -> e != p && e.isAlive() && e.isPickable()
                                && !(e instanceof ArmorStandEntity)
                                && !(e instanceof PlayerEntity && (((PlayerEntity) e).isCreative()
                                    || ((PlayerEntity) e).isSpectator())))
        );
        if (!nearby.contains(target)) nearby.add(target);
        nearby.sort((a, b) -> Double.compare(horizontalDistance(p, a), horizontalDistance(p, b)));
        if (nearby.size() > 8) {
            java.util.ArrayList<LivingEntity> nearest = new java.util.ArrayList<>(nearby.subList(0, 8));
            if (!nearest.contains(target)) {
                nearest.set(nearest.size() - 1, target);
                nearest.sort((a, b) -> Double.compare(horizontalDistance(p, a), horizontalDistance(p, b)));
            }
            nearby = nearest;
        }

        int close = 0;
        double distanceSum = 0.0;
        double healthSum = 0.0;
        double nearestOtherDistance = Double.MAX_VALUE;
        double nearestOtherDx = 0.0;
        double nearestOtherDz = 0.0;
        double pressureXSum = 0.0;
        double pressureZSum = 0.0;
        double pressureWeightSum = 0.0;
        for (LivingEntity enemy : nearby) {
            double distance = horizontalDistance(p, enemy);
            if (distance <= 3.25) close++;
            distanceSum += distance;
            healthSum += enemy.getHealth() + enemy.getAbsorptionAmount();
            if (enemy != target) {
                double dx = enemy.getX() - p.getX();
                double dz = enemy.getZ() - p.getZ();
                double weight = 1.0 / (1.0 + distance);
                double safeDistance = Math.max(distance, 1.0e-6);
                pressureXSum += dx / safeDistance * weight;
                pressureZSum += dz / safeDistance * weight;
                pressureWeightSum += weight;
                if (distance < nearestOtherDistance) {
                    nearestOtherDistance = distance;
                    nearestOtherDx = dx;
                    nearestOtherDz = dz;
                }
            }
        }
        int count = nearby.size();
        lastObservation[44] = count / 8.0F;
        lastObservation[45] = close / 8.0F;
        lastObservation[46] = count == 0 ? 0.0F
                : clip((float) (distanceSum / count / 20.0), 0.0F, 1.0F);
        lastObservation[47] = count == 0 ? 0.0F
                : clip((float) (healthSum / count / 20.0), 0.0F, 1.5F);
        lastObservation[48] = clip((float) (nearestOtherDx / 20.0), -1.0F, 1.0F);
        lastObservation[49] = clip((float) (nearestOtherDz / 20.0), -1.0F, 1.0F);
        lastObservation[50] = pressureWeightSum == 0.0 ? 0.0F
                : clip((float) (pressureXSum / pressureWeightSum), -1.0F, 1.0F);
        lastObservation[51] = pressureWeightSum == 0.0 ? 0.0F
                : clip((float) (pressureZSum / pressureWeightSum), -1.0F, 1.0F);
    }

    private static final class TerrainScan {
        final float[] clearance = new float[4];
        final float[] support = new float[4];
        final double[] directionX = new double[4];
        final double[] directionZ = new double[4];
        float targetHeightDelta;
        float forwardSlope;
    }

    /** Probe the same four approach directions used by the terrain-trained actor. */
    private static TerrainScan getTerrainScan(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        if (cachedTerrainScan != null && cachedTerrainTick == p.tickCount
                && cachedTerrainTargetId == t.getId()) return cachedTerrainScan;

        TerrainScan scan = new TerrainScan();
        double dx = t.getX() - p.getX();
        double dz = t.getZ() - p.getZ();
        double distance = Math.sqrt(dx * dx + dz * dz);
        double forwardX;
        double forwardZ;
        if (distance > 1.0e-6) {
            forwardX = dx / distance;
            forwardZ = dz / distance;
        } else {
            double yaw = Math.toRadians(p.yRot);
            forwardX = -Math.sin(yaw);
            forwardZ = Math.cos(yaw);
        }
        double rightX = -forwardZ;
        double rightZ = forwardX;
        scan.directionX[0] = forwardX;
        scan.directionZ[0] = forwardZ;
        scan.directionX[1] = -rightX;
        scan.directionZ[1] = -rightZ;
        scan.directionX[2] = rightX;
        scan.directionZ[2] = rightZ;
        scan.directionX[3] = -forwardX;
        scan.directionZ[3] = -forwardZ;

        for (int direction = 0; direction < 4; direction++) {
            float clearance = 1.0F;
            for (double probe : terrainSampleDistances) {
                double x = p.getX() + scan.directionX[direction] * probe;
                double z = p.getZ() + scan.directionZ[direction] * probe;
                if (blocksMovementAt(mc, x, p.getY(), z)
                        || !hasGroundSupport(mc, x, z, p.getY())) {
                    clearance = (float) (probe / 4.0);
                    break;
                }
            }
            scan.clearance[direction] = clearance;
            double supportX = p.getX() + scan.directionX[direction] * 1.5;
            double supportZ = p.getZ() + scan.directionZ[direction] * 1.5;
            scan.support[direction] = hasGroundSupport(mc, supportX, supportZ, p.getY()) ? 1.0F : 0.0F;
        }

        scan.targetHeightDelta = clip((float) ((t.getY() - p.getY()) / 3.0), -1.0F, 1.0F);
        double playerGround = groundHeight(mc, p.getX(), p.getZ(), p.getY());
        double aheadGround = groundHeight(mc, p.getX() + forwardX * 1.5,
                p.getZ() + forwardZ * 1.5, p.getY());
        if (Double.isFinite(playerGround) && Double.isFinite(aheadGround)) {
            scan.forwardSlope = clip((float) ((aheadGround - playerGround) / 2.0), -1.0F, 1.0F);
        }
        cachedTerrainScan = scan;
        cachedTerrainTick = p.tickCount;
        cachedTerrainTargetId = t.getId();
        return scan;
    }

    private static double groundHeight(Minecraft mc, double x, double z, double feetY) {
        if (mc.level == null) return Double.NaN;
        int blockX = (int) Math.floor(x);
        int blockZ = (int) Math.floor(z);
        int baseY = (int) Math.floor(feetY);
        // Search downward from feet level; looking above it would confuse cave ceilings with ground.
        int topY = baseY;
        int bottomY = Math.max(0, baseY - 8);
        for (int y = topY; y >= bottomY; y--) {
            BlockPos pos = new BlockPos(blockX, y, blockZ);
            if (mc.level.getBlockState(pos).getMaterial().blocksMotion()) return y + 1.0;
        }
        return Double.NaN;
    }

    private static boolean hasGroundSupport(Minecraft mc, double x, double z, double feetY) {
        double ground = groundHeight(mc, x, z, feetY);
        if (!Double.isFinite(ground)) return false;
        double delta = feetY - ground;
        // Permit a one-block step down/up, reject holes and steep drops.
        return delta >= -1.05 && delta <= 1.35;
    }

    private static boolean blocksMovementAt(Minecraft mc, double x, double feetY, double z) {
        if (mc.level == null) return true;
        int blockX = (int) Math.floor(x);
        int blockZ = (int) Math.floor(z);
        int y = (int) Math.floor(feetY + 0.05);
        BlockPos feet = new BlockPos(blockX, y, blockZ);
        BlockPos head = new BlockPos(blockX, y + 1, blockZ);
        return mc.level.getBlockState(feet).getMaterial().blocksMotion()
                || mc.level.getBlockState(head).getMaterial().blocksMotion();
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

    private static int countDetonatingMinecartKits(PlayerEntity player) {
        int powerSources = countInventoryItem(player, Items.REDSTONE_BLOCK)
                + countInventoryItem(player, Items.REDSTONE_TORCH);
        return Math.min(countInventoryItem(player, Items.TNT_MINECART),
                Math.min(countInventoryItem(player, Items.ACTIVATOR_RAIL), powerSources));
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
            if (isBuildingBlock(stack, false)) count += stack.getCount();
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

    private static int selectTacticalAction(ClientPlayerEntity p, LivingEntity t, int proposed) {
        if (isLongItemUse(p)) return proposed;
        int closeCount = Math.round(lastObservation[45] * 8.0F);
        double distance = horizontalDistance(p, t);
        float healthRatio = p.getHealth() / Math.max(1.0F, p.getMaxHealth());
        boolean recentDamage = p.tickCount - lastPlayerDamageTick <= 32;
        boolean hasHealPotion = findPotionInventorySlot(p, Effects.HEAL, true) >= 0;
        boolean hasApple = countHealingApples(p) > 0;
        boolean hasPearl = countInventoryItem(p, Items.ENDER_PEARL) > 0;
        boolean hasShield = countInventoryItem(p, Items.SHIELD) > 0;
        boolean hasCage = countInventoryItem(p, Items.OBSIDIAN) >= 4;
        boolean hasBuildingBlocks = findInventoryBlockSlot(p, false) >= 0;

        // Escape an active chain before choosing damage or utility actions.
        if (playerComboTaken > 0 && recentDamage && hasComboEscape(p)) return 28;

        // Conserve healing until it can make a meaningful difference; at critical
        // health, the override is stronger than the policy's original action.
        float missingHealth = p.getMaxHealth() - p.getHealth();
        if (healthRatio <= 0.40F && hasHealPotion && missingHealth >= 4.0F) return 14;
        if (healthRatio <= 0.62F && hasApple
                && (missingHealth >= 5.0F || (p.getAbsorptionAmount() < 1.5F && closeCount >= 2))) {
            return 13;
        }
        if (healthRatio <= 0.55F && hasHealPotion && missingHealth >= 5.0F) return 14;

        // When surrounded at low health, prefer escape/defense over a forward trade.
        if (closeCount >= 2 && healthRatio <= 0.58F) {
            if (hasPearl) return 15;
            if (hasShield) return 12;
            if (hasCage) return 25;
            if (hasBuildingBlocks) return 18;
            return 2;
        }
        if (recentDamage && closeCount >= 2 && distance <= 4.0 && hasShield) return 12;

        // Keep pressure on a target already reeling: a charged follow-up uses the
        // learned W-tap/sprint-reset action instead of waiting for another random pick.
        if (enemyComboTaken > 0 && p.tickCount - lastEnemyDamageTick <= 22
                && distance <= 3.45 && p.getAttackStrengthScale(0.5F) >= 0.95F) return 26;

        // Schedule a clean jump-crit only when there is room to swing, not while
        // charging into a packed group.
        boolean attackIntent = proposed == 5 || proposed == 6 || proposed == 7 || proposed == 11;
        if (attackIntent && distance <= 3.25 && closeCount <= 1 && p.isOnGround()
                && p.getAttackStrengthScale(0.5F) >= 0.95F
                && p.tickCount - lastCritAttackTick >= 32) return 9;

        // If the learner is outside reach while at least two enemies are already
        // nearby, back out and re-approach from the edge rather than pushing inward.
        boolean forwardIntent = proposed == 1 || proposed == 5 || proposed == 9 || proposed == 10;
        if (closeCount >= 2 && (proposed == 0 || proposed == 8 || proposed == 9
                || (distance > 3.25 && forwardIntent))) return 2;
        return proposed;
    }

    private static boolean hasComboEscape(ClientPlayerEntity p) {
        return countInventoryItem(p, Items.SHIELD) > 0
                || countInventoryItem(p, Items.WATER_BUCKET) > 0
                || countInventoryItem(p, Items.ENDER_PEARL) > 0;
    }

    private static void applyPolicyAction(Minecraft mc, ClientPlayerEntity p, LivingEntity t, int action) {
        if (pendingCritTargetId >= 0 && (action == 2 || action == 12 || action == 13
                || action == 14 || action == 15 || action == 18 || action == 19
                || action == 20 || action == 21 || action == 22 || action == 23
                || action == 24 || action == 25 || action == 27 || action == 28 || action == 29)) {
            clearPendingCrit();
        }
        if (shieldActionActive && action != 12 && action != 28) {
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
            case 5: case 6: case 7: case 11:
                attackTarget(mc, p, t);
                break;
            case 9:
                queueCriticalJump(p, t);
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
            case 15: {
                boolean pearlEscape = (playerComboTaken > 0)
                        || (p.getHealth() / Math.max(1.0F, p.getMaxHealth()) <= 0.60F
                            && Math.round(lastObservation[45] * 8.0F) >= 2);
                if (useEnderPearl(mc, p, t, pearlEscape) && pearlEscape) {
                    scheduleRetreat(p, t, 10);
                }
                break;
            }
            case 16:
                if (!usePotion(mc, p, Effects.DAMAGE_BOOST, false)) {
                    usePotion(mc, p, Effects.MOVEMENT_SPEED, false);
                }
                break;
            case 17:
                useRangedItem(mc, p, t);
                break;
            case 18:
                placeBarricade(mc, p, t);
                break;
            case 19:
                placeTrap(mc, p, t);
                break;
            case 20:
                useWaterBucket(mc, p, t);
                break;
            case 21:
                placePillarBlock(mc, p);
                break;
            case 22:
                if (useCrystalCombo(mc, p, t, false)) scheduleRetreat(p, t, 18);
                break;
            case 23:
                if (useRespawnAnchor(mc, p, t)) scheduleRetreat(p, t, 18);
                break;
            case 24:
                if (useTntMinecart(mc, p, t)) scheduleRetreat(p, t, 60);
                break;
            case 25:
                placeObsidianCage(mc, p);
                break;
            case 26:
                inSprint = false;
                p.setSprinting(false);
                attackTarget(mc, p, t);
                break;
            case 27:
                if (useCrystalCombo(mc, p, t, true)) scheduleRetreat(p, t, 18);
                break;
            case 28:
                breakCombo(mc, p, t);
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

    private static boolean hasArrowAmmo(PlayerEntity p) {
        return countInventoryItem(p, Items.ARROW) + countInventoryItem(p, Items.TIPPED_ARROW)
                + countInventoryItem(p, Items.SPECTRAL_ARROW) > 0;
    }

    private static boolean isCrossbowCharged(ItemStack stack) {
        return stack != null && !stack.isEmpty() && stack.getItem() == Items.CROSSBOW
                && stack.hasTag() && stack.getTag().getBoolean("Charged");
    }

    private static int findCrossbowInventorySlot(PlayerEntity p, boolean chargedOnly) {
        if (p == null) return -1;
        for (int i = 0; i < p.inventory.items.size(); i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (stack.getItem() == Items.CROSSBOW
                    && (!chargedOnly || isCrossbowCharged(stack))) return i;
        }
        return -1;
    }

    private static int findHotbarCrossbow(ClientPlayerEntity p, boolean chargedOnly) {
        int inventorySlot = findCrossbowInventorySlot(p, chargedOnly);
        if (inventorySlot < 0) return -1;
        return inventorySlot < 9 ? inventorySlot : moveInventorySlotToHotbar(p, inventorySlot);
    }

    private static boolean hasUsableRangedAttack(ClientPlayerEntity p) {
        return findPotionInventorySlot(p, Effects.HARM, true) >= 0
                || countInventoryItem(p, Items.SNOWBALL) > 0
                || countInventoryItem(p, Items.EGG) > 0
                || findCrossbowInventorySlot(p, true) >= 0
                || (hasArrowAmmo(p) && (countInventoryItem(p, Items.CROSSBOW) > 0
                    || countInventoryItem(p, Items.BOW) > 0));
    }

    private static boolean hasHotbarResource(ClientPlayerEntity p, int action) {
        switch (action) {
            case 13:
                return countHealingApples(p) > 0;
            case 14: return findPotionInventorySlot(p, Effects.HEAL, true) >= 0;
            case 15: return countInventoryItem(p, Items.ENDER_PEARL) > 0;
            case 16: return findPotionInventorySlot(p, Effects.DAMAGE_BOOST, false) >= 0
                        || findPotionInventorySlot(p, Effects.MOVEMENT_SPEED, false) >= 0;
            case 17: return hasUsableRangedAttack(p);
            case 18: case 21: return findInventoryBlockSlot(p, false) >= 0;
            case 19: return countInventoryItem(p, Items.COBWEB) > 0
                        || countInventoryItem(p, Items.LAVA_BUCKET) > 0;
            case 20: return countInventoryItem(p, Items.WATER_BUCKET) > 0;
            case 22: return countInventoryItem(p, Items.END_CRYSTAL) > 0
                        && countInventoryItem(p, Items.OBSIDIAN) > 0;
            case 23:
                return countInventoryItem(p, Items.RESPAWN_ANCHOR) > 0
                        && countInventoryItem(p, Items.GLOWSTONE) > 0
                        && !p.level.dimensionType().respawnAnchorWorks();
            case 24: return countDetonatingMinecartKits(p) > 0;
            case 25: return countInventoryItem(p, Items.OBSIDIAN) >= 4;
            case 27: return countInventoryItem(p, Items.ENDER_PEARL) > 0
                        && countInventoryItem(p, Items.END_CRYSTAL) > 0
                        && countInventoryItem(p, Items.OBSIDIAN) > 0;
            default: return false;
        }
    }

    private static void applyNavigationAssist(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        double distance = horizontalDistance(p, t);
        TerrainScan scan = getTerrainScan(mc, p, t);
        float aggression = AiPilotSettings.aggression();
        float pursuit = AiPilotSettings.pursuit();
        float strafing = AiPilotSettings.strafing();
        float caution = AiPilotSettings.caution();

        boolean blockedAhead = scan.clearance[0] < 0.99F || scan.support[0] < 0.5F;
        boolean deliberateRetreat = lastAction == 2 || lastAction == 11 || lastAction == 12
                || lastAction == 18 || lastAction == 22 || lastAction == 23
                || lastAction == 24 || lastAction == 28;
        boolean preservePolicyMovement = deliberateRetreat || lastAction == 25 || lastAction == 8
                || lastAction == 3 || lastAction == 4 || lastAction == 6 || lastAction == 7;
        if (deliberateRetreat && distance > 1.65) {
            int retreatDirection = 3;
            if (scan.support[retreatDirection] <= 0.5F || scan.clearance[retreatDirection] <= 0.25F) {
                retreatDirection = -1;
                for (int candidate : new int[]{1, 2}) {
                    if (scan.support[candidate] > 0.5F && scan.clearance[candidate] > 0.25F
                            && (retreatDirection < 0
                                || scan.clearance[candidate] > scan.clearance[retreatDirection])) {
                        retreatDirection = candidate;
                    }
                }
            }
            if (retreatDirection >= 0) {
                setMovementToward(p, scan.directionX[retreatDirection],
                        scan.directionZ[retreatDirection], 0.82F);
            } else {
                inForward = 0.0F;
                inStrafe = 0.0F;
                inSprint = false;
            }
        } else if (preservePolicyMovement) {
            // Respect back/strafe/guard/cage intents instead of silently pushing forward.
        } else if (blockedAhead && distance > 1.65 && pursuit > 0.01F) {
            double[] route = chooseSafePursuitDirection(mc, p, scan);
            float strength = 0.45F + 0.55F * pursuit;
            setMovementToward(p, route[0], route[1], strength);
            if (p.isOnGround() && scan.support[0] > 0.5F
                    && isLowObstacleAhead(mc, p, scan.directionX[0], scan.directionZ[0])) {
                inJump = true;
            }
        } else if (distance > 3.0 && pursuit > 0.01F) {
            // Pursuit assist must not cancel a PPO-selected retreat/strafe/guard.
            float chase = (0.18F + 0.72F * aggression) * pursuit;
            inForward = Math.max(inForward, chase);
            inSprint = aggression > 0.45F && distance > 5.0;
        }

        if (distance >= 1.45 && distance <= 4.0 && strafing > 0.01F) {
            boolean leftSafe = scan.clearance[1] > 0.25F && scan.support[1] > 0.5F;
            boolean rightSafe = scan.clearance[2] > 0.25F && scan.support[2] > 0.5F;
            if (leftSafe || rightSafe) {
                int phase = (((p.tickCount / 20) + t.getId()) & 1) == 0 ? 1 : -1;
                if (leftSafe && !rightSafe) phase = 1;
                if (rightSafe && !leftSafe) phase = -1;
                inStrafe = clip(inStrafe + phase * strafing * 0.28F, -1.0F, 1.0F);
            }
        }

        float maxHealth = Math.max(1.0F, p.getMaxHealth());
        float healthRatio = (p.getHealth() + p.getAbsorptionAmount()) / maxHealth;
        float retreatThreshold = 0.10F + 0.30F * caution;
        if (caution > 0.01F && healthRatio < retreatThreshold && distance < 4.0) {
            inForward = Math.min(inForward, -0.20F - 0.45F * caution);
        }

        if (pendingCritTargetId == t.getId() && p.tickCount <= pendingCritUntilTick) {
            inSprint = false;
            if (p.isOnGround()) {
                inForward = Math.max(inForward, 0.65F);
                inJump = true;
            } else {
                inForward = 0.0F;
                inStrafe = 0.0F;
                inJump = false;
            }
        }

        if (forcedRetreatTargetId == t.getId() && p.tickCount <= forcedRetreatUntilTick) {
            int retreatDirection = 3;
            if (scan.support[retreatDirection] <= 0.5F || scan.clearance[retreatDirection] <= 0.25F) {
                retreatDirection = -1;
                for (int candidate : new int[]{1, 2}) {
                    if (scan.support[candidate] > 0.5F && scan.clearance[candidate] > 0.25F
                            && (retreatDirection < 0
                                || scan.clearance[candidate] > scan.clearance[retreatDirection])) {
                        retreatDirection = candidate;
                    }
                }
            }
            if (retreatDirection >= 0) {
                setMovementToward(p, scan.directionX[retreatDirection],
                        scan.directionZ[retreatDirection], 0.95F);
            } else {
                inForward = 0.0F;
                inStrafe = 0.0F;
                inSprint = false;
            }
            inJump = false;
        } else if (p.tickCount > forcedRetreatUntilTick) {
            forcedRetreatUntilTick = -1;
            forcedRetreatTargetId = -1;
        }
    }

    private static double[] chooseSafePursuitDirection(
            Minecraft mc, ClientPlayerEntity p, TerrainScan scan) {
        double targetX = scan.directionX[0];
        double targetZ = scan.directionZ[0];
        double rightX = -targetZ;
        double rightZ = targetX;
        double[] angles = {0.0, -35.0, 35.0, -60.0, 60.0, -90.0, 90.0, 180.0};
        double bestScore = -Double.MAX_VALUE;
        double bestX = targetX;
        double bestZ = targetZ;
        for (double angle : angles) {
            double radians = Math.toRadians(angle);
            double dirX = targetX * Math.cos(radians) + rightX * Math.sin(radians);
            double dirZ = targetZ * Math.cos(radians) + rightZ * Math.sin(radians);
            double clearance = 1.0;
            for (double probe : terrainSampleDistances) {
                double x = p.getX() + dirX * probe;
                double z = p.getZ() + dirZ * probe;
                if (blocksMovementAt(mc, x, p.getY(), z)
                        || !hasGroundSupport(mc, x, z, p.getY())) {
                    clearance = probe / 4.0;
                    break;
                }
            }
            boolean supported = hasGroundSupport(mc, p.getX() + dirX * 1.5,
                    p.getZ() + dirZ * 1.5, p.getY());
            double progress = dirX * targetX + dirZ * targetZ;
            double score = 0.55 * progress + 0.45 * clearance + (supported ? 0.25 : -1.5);
            if (score > bestScore) {
                bestScore = score;
                bestX = dirX;
                bestZ = dirZ;
            }
        }
        return new double[]{bestX, bestZ};
    }

    private static void setMovementToward(ClientPlayerEntity p, double directionX,
                                          double directionZ, float strength) {
        double yaw = Math.toRadians(p.yRot);
        double forwardX = -Math.sin(yaw);
        double forwardZ = Math.cos(yaw);
        double rightX = Math.cos(yaw);
        double rightZ = Math.sin(yaw);
        inForward = clip((float) ((directionX * forwardX + directionZ * forwardZ) * strength), -1.0F, 1.0F);
        // MovementInput.leftImpulse is positive for left; the local right vector is opposite.
        inStrafe = clip((float) (-(directionX * rightX + directionZ * rightZ) * strength), -1.0F, 1.0F);
        inSprint = strength > 0.75F;
    }

    private static boolean isLowObstacleAhead(Minecraft mc, ClientPlayerEntity p,
                                               double directionX, double directionZ) {
        double x = p.getX() + directionX * 0.75;
        double z = p.getZ() + directionZ * 0.75;
        int blockX = (int) Math.floor(x);
        int blockZ = (int) Math.floor(z);
        int feetY = (int) Math.floor(p.getY() + 0.05);
        boolean feetBlocked = mc.level.getBlockState(new BlockPos(blockX, feetY, blockZ))
                .getMaterial().blocksMotion();
        boolean headBlocked = mc.level.getBlockState(new BlockPos(blockX, feetY + 1, blockZ))
                .getMaterial().blocksMotion();
        return feetBlocked && !headBlocked;
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
            case 9: inForward = 1.0F; break;
            case 10: inForward = 1.0F; inSprint = true; break;
            case 11: inForward = -1.0F; break;
            case 12: inForward = -0.45F; break;
            case 18: case 22: case 23: case 28: inForward = -0.45F; break;
            case 21: inJump = true; break;
            case 24: inForward = -0.8F; inSprint = true; break;
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

    private static void queueCriticalJump(ClientPlayerEntity p, LivingEntity t) {
        if (t == null || !t.isAlive() || p.isUsingItem() || !p.isOnGround()) return;
        if (pendingCritTargetId < 0 && p.tickCount - lastCritAttackTick < 32) return;
        if (pendingCritTargetId != t.getId() || p.tickCount > pendingCritUntilTick) {
            pendingCritTargetId = t.getId();
            pendingCritUntilTick = p.tickCount + 16; // same 0.8 s window as the simulator
        }
        inSprint = false;
        p.setSprinting(false);
        inForward = 1.0F;
        inJump = true;
    }

    private static void processPendingCrit(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        if (pendingCritTargetId < 0) return;
        if (t == null || !t.isAlive() || t.getId() != pendingCritTargetId
                || p.tickCount > pendingCritUntilTick) {
            clearPendingCrit();
            return;
        }
        if (isLongItemUse(p) || p.isUsingItem()) return;
        if (p.isOnGround() || p.getDeltaMovement().y >= -0.05) return;

        // Vanilla critical hits require a clean fall, no sprint, and no water/passenger state.
        inSprint = false;
        p.setSprinting(false);
        if (p.isInWater() || p.isPassenger() || horizontalDistance(p, t) > 3.45
                || !p.canSee(t) || p.getAttackStrengthScale(0.5F) < 0.95F) return;

        mc.gameMode.attack(p, t);
        p.swing(Hand.MAIN_HAND);
        lastMeleeAttackTick = p.tickCount;
        lastCritAttackTick = p.tickCount;
        enemyStunUntilTick = p.tickCount + 9;
        clearPendingCrit();
    }

    private static void clearPendingCrit() {
        pendingCritTargetId = -1;
        pendingCritUntilTick = -1;
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

    private static boolean raisePpoShield(Minecraft mc, ClientPlayerEntity p) {
        if (p.isBlocking()) {
            shieldActionActive = true;
            return true;
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
        if (hand == null) return false;
        mc.gameMode.useItem(p, p.level, hand);
        shieldActionActive = true;
        return true;
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

    private static boolean useEnderPearl(Minecraft mc, ClientPlayerEntity p,
                                         LivingEntity t, boolean escape) {
        int slot = findHotbarSlot(p, Items.ENDER_PEARL);
        if (slot < 0 || p.isUsingItem()) return false;
        if (escape && t != null) {
            double dx = p.getX() - t.getX();
            double dz = p.getZ() - t.getZ();
            float yaw = (float) Math.toDegrees(Math.atan2(dz, dx)) - 90.0F;
            forceAim(p, yaw, -8.0F, 4);
        }
        select(p, slot);
        mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
        return true;
    }

    private static boolean usePotion(Minecraft mc, ClientPlayerEntity p, Effect effect, boolean splashOnly) {
        int slot = findHotbarPotion(p, effect, splashOnly);
        if (slot < 0 || p.isUsingItem()) return false;
        if (splashOnly && effect == Effects.HEAL) {
            forceAim(p, p.yRot, 85.0F, 3); // splash the healing potion at the player's feet
        }
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
        int inventorySlot = findPotionInventorySlot(p, effect, splashOnly);
        return inventorySlot < 0 ? -1 : moveInventorySlotToHotbar(p, inventorySlot);
    }

    private static int findPotionInventorySlot(PlayerEntity p, Effect effect, boolean splashOnly) {
        if (p == null || effect == null) return -1;
        for (int i = 0; i < p.inventory.items.size(); i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (!(stack.getItem() instanceof net.minecraft.item.PotionItem)) continue;
            if (splashOnly && !(stack.getItem() instanceof SplashPotionItem)) continue;
            if (!splashOnly && stack.getItem() instanceof SplashPotionItem) continue;
            if (hasPotionEffect(stack, effect)) return i;
        }
        return -1;
    }

    private static void useRangedItem(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        if (t == null || !t.isAlive() || p.isUsingItem() || !p.canSee(t)) return;
        double distance = horizontalDistance(p, t);
        if (distance < 1.6 || distance > 24.0) return;

        // Fire a preloaded crossbow immediately; otherwise load a crossbow or draw a bow.
        int crossbowSlot = findHotbarCrossbow(p, true);
        if (crossbowSlot >= 0) {
            aimAtProjectileTarget(p, t);
            select(p, crossbowSlot);
            mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
            return;
        }
        if (hasArrowAmmo(p)) {
            crossbowSlot = findHotbarCrossbow(p, false);
            if (crossbowSlot >= 0) {
                ItemStack crossbow = p.inventory.items.get(crossbowSlot);
                if (isCrossbowCharged(crossbow)) {
                    aimAtProjectileTarget(p, t);
                    select(p, crossbowSlot);
                    mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
                } else {
                    aimAtProjectileTarget(p, t);
                    select(p, crossbowSlot);
                    mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
                    rangedReleaseTick = p.tickCount + 25;
                }
                return;
            }

            int bowSlot = findHotbarSlot(p, Items.BOW);
            if (bowSlot >= 0) {
                aimAtProjectileTarget(p, t);
                select(p, bowSlot);
                mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
                rangedReleaseTick = p.tickCount + 20;
                return;
            }
        }

        int harmingPotion = findHotbarPotion(p, Effects.HARM, true);
        if (harmingPotion >= 0) {
            select(p, harmingPotion);
            mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
            return;
        }
        useHotbarItem(mc, p, Items.SNOWBALL, Items.EGG);
    }

    private static int findHotbarSlot(PlayerEntity p, Item item) {
        for (int i = 0; i < 9; i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (!stack.isEmpty() && stack.getItem() == item) return i;
        }
        int inventorySlot = findInventorySlot(p, item);
        if (inventorySlot < 9) return -1;
        return p instanceof ClientPlayerEntity
                ? moveInventorySlotToHotbar((ClientPlayerEntity) p, inventorySlot) : -1;
    }

    private static int findInventorySlot(PlayerEntity p, Item item) {
        if (p == null || item == null) return -1;
        for (int i = 0; i < p.inventory.items.size(); i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (!stack.isEmpty() && stack.getItem() == item) return i;
        }
        return -1;
    }

    /** Move an inventory item with the vanilla SWAP click; keep the combat weapon intact. */
    private static int moveInventorySlotToHotbar(ClientPlayerEntity p, int inventorySlot) {
        Minecraft mc = Minecraft.getInstance();
        if (p == null || mc.gameMode == null || p.isUsingItem()
                || inventorySlot < 9 || inventorySlot >= p.inventory.items.size()
                || p.containerMenu == null || p.containerMenu.containerId != 0) return -1;

        int hotbarSlot = -1;
        for (int i = 0; i < 9; i++) {
            if (i != combatWeaponSlot && p.inventory.items.get(i).isEmpty()) {
                hotbarSlot = i;
                break;
            }
        }
        if (hotbarSlot < 0) {
            for (int i = 0; i < 9; i++) {
                if (i != combatWeaponSlot && i != p.inventory.selected) {
                    hotbarSlot = i;
                    break;
                }
            }
        }
        if (hotbarSlot < 0) {
            for (int i = 0; i < 9; i++) {
                if (i != combatWeaponSlot) {
                    hotbarSlot = i;
                    break;
                }
            }
        }
        if (hotbarSlot < 0) return -1;

        mc.gameMode.handleInventoryMouseClick(
                p.containerMenu.containerId, inventorySlot, hotbarSlot, ClickType.SWAP, p);
        return hotbarSlot;
    }

    private static int findHotbarBlock(PlayerEntity p, boolean obsidianOnly) {
        for (int i = 0; i < 9; i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (isBuildingBlock(stack, obsidianOnly)) return i;
        }
        int inventorySlot = findInventoryBlockSlot(p, obsidianOnly);
        if (inventorySlot < 9) return -1;
        return p instanceof ClientPlayerEntity
                ? moveInventorySlotToHotbar((ClientPlayerEntity) p, inventorySlot) : -1;
    }

    private static int findInventoryBlockSlot(PlayerEntity p, boolean obsidianOnly) {
        if (p == null) return -1;
        for (int i = 0; i < p.inventory.items.size(); i++) {
            if (isBuildingBlock(p.inventory.items.get(i), obsidianOnly)) return i;
        }
        return -1;
    }

    private static boolean isBuildingBlock(ItemStack stack, boolean obsidianOnly) {
        if (stack == null || stack.isEmpty() || !(stack.getItem() instanceof BlockItem)) return false;
        Item item = stack.getItem();
        if (obsidianOnly) return item == Items.OBSIDIAN;
        if (item == Items.OBSIDIAN || item == Items.COBWEB || item == Items.RESPAWN_ANCHOR
                || item == Items.REDSTONE_BLOCK || item == Items.GLOWSTONE) return false;
        return ((BlockItem) item).getBlock().defaultBlockState().getMaterial().blocksMotion();
    }

    private static boolean placeBlockAt(Minecraft mc, ClientPlayerEntity p, BlockPos desired, Item item) {
        if (mc.level == null || !mc.level.getBlockState(desired).isAir()) return false;
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

    private static boolean useBucketAt(Minecraft mc, ClientPlayerEntity p, Item bucket, BlockPos desired) {
        int slot = findHotbarSlot(p, bucket);
        if (slot < 0 || p.isUsingItem()) return false;
        BlockPos support = desired.below();
        if (mc.level.getBlockState(support).isAir()) return false;
        select(p, slot);
        useOnBlock(mc, p, support, Direction.UP);
        return true;
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

    private static boolean useWaterBucket(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        double dist = horizontalDistance(p, t);
        boolean hazard = p.isOnFire()
                || mc.level.getBlockState(p.blockPosition()).getBlock() == Blocks.LAVA
                || mc.level.getBlockState(p.blockPosition()).getBlock() == Blocks.COBWEB;
        if (dist <= 3.4) {
            return useBucketAt(mc, p, Items.WATER_BUCKET, targetAdjacentBlock(mc, p, t));
        }
        return hazard && useBucketAt(mc, p, Items.WATER_BUCKET, p.blockPosition());
    }

    private static boolean useWaterBucket(Minecraft mc, ClientPlayerEntity p) {
        return useBucketAt(mc, p, Items.WATER_BUCKET, p.blockPosition());
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

    private static boolean useCrystalCombo(Minecraft mc, ClientPlayerEntity p,
                                           LivingEntity t, boolean pearlFirst) {
        double dist = horizontalDistance(p, t);
        if (pearlFirst) {
            if (dist < 3.6 || dist > 12.0 || findHotbarSlot(p, Items.ENDER_PEARL) < 0
                    || findHotbarSlot(p, Items.END_CRYSTAL) < 0
                    || findHotbarSlot(p, Items.OBSIDIAN) < 0) return false;
            if (!useHotbarItem(mc, p, Items.ENDER_PEARL)) return false;
        } else if (dist < 2.4 || dist > 5.5
                || findHotbarSlot(p, Items.END_CRYSTAL) < 0
                || findHotbarSlot(p, Items.OBSIDIAN) < 0) {
            return false;
        }
        BlockPos base = targetAdjacentBlock(mc, p, t);
        if (!placeBlockAt(mc, p, base, Items.OBSIDIAN)) return false;
        int crystalSlot = findHotbarSlot(p, Items.END_CRYSTAL);
        if (crystalSlot < 0) return false;
        select(p, crystalSlot);
        useOnBlock(mc, p, base, Direction.UP);
        placedCrystalBase = base;
        crystalDetonateUntilTick = p.tickCount + 30;
        return true;
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

    private static boolean useRespawnAnchor(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        double dist = horizontalDistance(p, t);
        if (dist < 1.5 || dist > 4.8 || p.level.dimensionType().respawnAnchorWorks()
                || countInventoryItem(p, Items.GLOWSTONE) <= 0
                || findHotbarSlot(p, Items.RESPAWN_ANCHOR) < 0) return false;
        BlockPos anchor = targetAdjacentBlock(mc, p, t);
        if (!placeBlockAt(mc, p, anchor, Items.RESPAWN_ANCHOR)) return false;
        int glowstone = findHotbarSlot(p, Items.GLOWSTONE);
        if (glowstone < 0) return false;
        select(p, glowstone);
        useOnBlock(mc, p, anchor, Direction.UP);
        if (combatWeaponSlot >= 0) select(p, combatWeaponSlot);
        useOnBlock(mc, p, anchor, Direction.UP);
        return true;
    }

    private static boolean useTntMinecart(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        double dist = horizontalDistance(p, t);
        if (dist < 4.0 || dist > 8.5 || countDetonatingMinecartKits(p) <= 0) return false;
        Item powerItem = countInventoryItem(p, Items.REDSTONE_BLOCK) > 0
                ? Items.REDSTONE_BLOCK : Items.REDSTONE_TORCH;
        int railSlot = findHotbarSlot(p, Items.ACTIVATOR_RAIL);
        int cartSlot = findHotbarSlot(p, Items.TNT_MINECART);
        int powerSlot = findHotbarSlot(p, powerItem);
        if (railSlot < 0 || cartSlot < 0 || powerSlot < 0) return false;

        BlockPos rail = targetAdjacentBlock(mc, p, t);
        BlockPos power = adjacentPowerPosition(mc, rail);
        if (power == null || !placeBlockAt(mc, p, rail, Items.ACTIVATOR_RAIL)) return false;

        cartSlot = findHotbarSlot(p, Items.TNT_MINECART);
        if (cartSlot < 0) return false;
        select(p, cartSlot);
        useOnBlock(mc, p, rail, Direction.UP);

        powerSlot = findHotbarSlot(p, powerItem);
        if (powerSlot < 0) return false;
        return placeBlockAt(mc, p, power, powerItem);
    }

    private static BlockPos adjacentPowerPosition(Minecraft mc, BlockPos rail) {
        Direction[] sides = {Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST};
        for (Direction side : sides) {
            BlockPos candidate = rail.relative(side);
            if (mc.level.getBlockState(candidate).isAir()
                    && mc.level.getBlockState(candidate.below()).getMaterial().blocksMotion()) {
                return candidate;
            }
        }
        return null;
    }

    private static void placeObsidianCage(Minecraft mc, ClientPlayerEntity p) {
        if (countInventoryItem(p, Items.OBSIDIAN) < 4) return;
        BlockPos center = p.blockPosition();
        BlockPos[] walls = {center.north(), center.south(), center.east(), center.west()};
        for (BlockPos wall : walls) placeBlockAt(mc, p, wall, Items.OBSIDIAN);
    }

    private static void breakCombo(Minecraft mc, ClientPlayerEntity p, LivingEntity t) {
        if (playerComboTaken <= 0 || p.tickCount - lastPlayerDamageTick > 32) return;
        boolean escaped = false;
        if (p.getOffhandItem().getItem() instanceof ShieldItem
                || p.getMainHandItem().getItem() instanceof ShieldItem
                || countInventoryItem(p, Items.SHIELD) > 0) {
            escaped = raisePpoShield(mc, p);
        }
        if (!escaped && countInventoryItem(p, Items.WATER_BUCKET) > 0) {
            escaped = useWaterBucket(mc, p);
        }
        if (!escaped && countInventoryItem(p, Items.ENDER_PEARL) > 0) {
            escaped = useEnderPearl(mc, p, t, true);
        }
        if (!escaped) return;
        playerComboTaken = 0;
        lastPlayerDamageTick = p.tickCount;
        counterUntilTick = p.tickCount + 36;
        counterArmedUntilTick = 0;
        scheduleRetreat(p, t, 12);
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
            double score = horizontalDistance(p, e)
                    + 0.15 * (e.getHealth() + e.getAbsorptionAmount());
            if (!p.canSee(e)) score += 4.0;
            if (e instanceof PlayerEntity) score -= 1.5;
            if (score < bestScore) { bestScore = score; best = e; }
            if (eye.distanceToSqr(e.position()) < 0.0001) continue;
        }
        return best;
    }

    // ------------------------------------------------------------------- aim
    private static void aimAtTargetOrForced(ClientPlayerEntity p, LivingEntity t, double dist) {
        if (p.tickCount <= forcedAimUntilTick) {
            p.yRot = forcedAimYaw;
            p.xRot = forcedAimPitch;
            p.yHeadRot = forcedAimYaw;
            p.yBodyRot = forcedAimYaw;
        } else {
            aimAt(p, t, dist);
        }
    }

    private static void forceAim(ClientPlayerEntity p, float yaw, float pitch, int holdTicks) {
        forcedAimYaw = yaw;
        forcedAimPitch = pitch;
        forcedAimUntilTick = p.tickCount + holdTicks;
        p.yRot = yaw;
        p.xRot = pitch;
        p.yHeadRot = yaw;
        p.yBodyRot = yaw;
    }

    private static void scheduleRetreat(ClientPlayerEntity p, LivingEntity t, int holdTicks) {
        if (t == null || holdTicks <= 0) return;
        forcedRetreatTargetId = t.getId();
        forcedRetreatUntilTick = p.tickCount + holdTicks;
    }

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

    /** Aim an arrow ahead of a moving target and compensate for modest projectile drop. */
    private static void aimAtProjectileTarget(ClientPlayerEntity p, LivingEntity t) {
        Vector3d targetVelocity = t.getDeltaMovement();
        double dx = t.getX() - p.getX();
        double dz = t.getZ() - p.getZ();
        double distance = Math.sqrt(dx * dx + dz * dz);
        double flightTicks = Math.min(15.0, distance / 3.0);
        double aimX = dx + targetVelocity.x * flightTicks;
        double aimZ = dz + targetVelocity.z * flightTicks;
        double aimY = (t.getY() + t.getBbHeight() * 0.55)
                - (p.getY() + p.getEyeHeight())
                + targetVelocity.y * flightTicks
                + 0.025 * flightTicks * (flightTicks + 1.0);
        double horizontal = Math.sqrt(aimX * aimX + aimZ * aimZ);
        float yaw = (float) Math.toDegrees(Math.atan2(aimZ, aimX)) - 90.0F;
        float pitch = (float) -Math.toDegrees(Math.atan2(aimY, horizontal));
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
        for (int i = 0; i < p.inventory.items.size(); i++) {
            ItemStack stack = p.inventory.items.get(i);
            if (stack.isEmpty()) continue;
            Item item = stack.getItem();
            if (!(item instanceof SwordItem || item instanceof AxeItem || item instanceof TridentItem)) continue;
            double damage = attackDamage(stack);
            if (damage > bestDmg) {
                bestDmg = damage;
                best = i;
            }
        }
        if (best >= 9) best = moveInventorySlotToHotbar(p, best);
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

package com.arena.aipilot;

import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.player.ClientPlayerEntity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.ai.attributes.AttributeModifier;
import net.minecraft.entity.ai.attributes.Attributes;
import net.minecraft.entity.item.ArmorStandEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.inventory.EquipmentSlotType;
import net.minecraft.item.AxeItem;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.item.ShieldItem;
import net.minecraft.item.SwordItem;
import net.minecraft.item.TridentItem;
import net.minecraft.util.Hand;
import net.minecraft.util.MovementInput;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.vector.Vector3d;
import net.minecraftforge.client.event.RenderGameOverlayEvent;

import java.util.List;
import java.util.Locale;

/**
 * Логика авто-боя. Ничего не отправляет в сеть напрямую — только вращает камеру,
 * жмёт инпуты и вызывает ванильные действия клиента (атака, использование предмета,
 * смена слота). Для сервера это неотличимо от обычного игрока.
 */
public final class AiPilotEngine {

    // ------------------------------------------------------------------ styles
    public enum Style {
        AGGRO("AGGRO", 3.0, 0.55, false),
        BALANCED("BALANCED", 3.0, 0.72, true),
        SAFE("SAFE", 3.0, 0.85, true);

        public final String label;
        public final double reach;
        public final double attackThreshold;
        public final boolean defensive;

        Style(String label, double reach, double attackThreshold, boolean defensive) {
            this.label = label;
            this.reach = reach;
            this.attackThreshold = attackThreshold;
            this.defensive = defensive;
        }
    }

    private static Style style = Style.BALANCED;
    private static double searchRange = 17.0;
    private static boolean autoCrit = true;
    private static boolean autoEat = true;
    private static boolean autoWeapon = true;

    // ------------------------------------------------------------------ state
    private static LivingEntity target;
    private static int targetId = -1;
    private static int escapeTicks = 0;
    private static int wTapTicks = 0;
    private static int eatSlot = -1;
    private static int stuckTicks = 0;
    private static double lastDist = 99.0;
    private static int attackingCooldown = 0;

    // Интенты движения, которые выставляет «мозг» и применяет InputUpdateEvent.
    private static float inForward = 0.0F;
    private static float inStrafe = 0.0F;
    private static boolean inJump = false;
    private static boolean inBack = false;
    private static boolean inSprint = false;

    private AiPilotEngine() {}

    public static void reset() {
        target = null;
        targetId = -1;
        escapeTicks = 0;
        wTapTicks = 0;
        eatSlot = -1;
        stuckTicks = 0;
        lastDist = 99.0;
        attackingCooldown = 0;
        inForward = 0.0F;
        inStrafe = 0.0F;
        inJump = false;
        inBack = false;
        inSprint = false;
    }

    public static void cycleStyle() {
        Style[] all = Style.values();
        style = all[(style.ordinal() + 1) % all.length];
        AiPilot.chat("\u00A7bAI Pilot: стиль " + style.label);
    }

    public static void setStyle(String name) {
        for (Style s : Style.values()) {
            if (s.label.equalsIgnoreCase(name)) {
                style = s;
                AiPilot.chat("\u00A7bAI Pilot: стиль " + s.label);
                return;
            }
        }
        AiPilot.chat("\u00A7cAI Pilot: неизвестный стиль " + name + " (AGGRO/BALANCED/SAFE)");
    }

    public static void setRange(double blocks) {
        searchRange = blocks;
        AiPilot.chat("\u00A7bAI Pilot: радиус поиска " + String.format(Locale.ROOT, "%.1f", blocks));
    }

    public static void setCrit(boolean on) {
        autoCrit = on;
        AiPilot.chat("\u00A7bAI Pilot: крит-прыжки " + (on ? "ВКЛ" : "ВЫКЛ"));
    }

    public static String statusLine() {
        ClientPlayerEntity p = Minecraft.getInstance().player;
        String hp = p == null ? "-" : String.format(Locale.ROOT, "%.1f", p.getHealth());
        return "\u00A7bAI Pilot \u00A77| \u00A7f" + (AiPilot.isEnabled() ? "ВКЛ" : "ВЫКЛ")
                + " \u00A77| стиль \u00A7f" + style.label
                + " \u00A77| HP \u00A7f" + hp
                + " \u00A77| цель \u00A7f" + (target == null ? "нет" : target.getName().getString())
                + " \u00A77| радиус \u00A7f" + String.format(Locale.ROOT, "%.1f", searchRange)
                + "\u00A77. \u00A78Клавиши: O — вкл/выкл, K — стиль, J — HUD";
    }

    // ------------------------------------------------------------------ brain
    /** Вызывается каждый тик игрока (Phase.START), до отправки позиции на сервер. */
    public static void tick(Minecraft mc) {
        ClientPlayerEntity p = mc.player;
        if (p == null || mc.level == null || mc.gameMode == null) return;
        if (mc.screen != null) {                 // открыт инвентарь/чат — не мешаем
            releaseKeys(mc);
            return;
        }
        if (p.isDeadOrDying()) {
            releaseKeys(mc);
            return;
        }

        // 1. выбор цели
        if (target == null || !target.isAlive() || target.level != p.level
                || p.distanceTo(target) > searchRange
                || (target instanceof PlayerEntity && (((PlayerEntity) target).isCreative()
                    || ((PlayerEntity) target).isSpectator()))) {
            target = findTarget(mc, p);
        }

        // 2. цель есть — ведём бой
        if (target != null) {
            double dist = p.distanceTo(target);
            escapeTicks = Math.max(0, escapeTicks - 1);
            if (p.hurtTime > 0 && style.defensive) {
                // нас бьют цепочкой — короткий выход из комбо (рывок назад + щит)
                escapeTicks = Math.max(escapeTicks, 5);
            }
            aimAt(p, target, dist);

            if (eatSlot >= 0) { eat(mc, p, eatSlot); return; }         // еду не прерываем

            if (autoEat && p.getHealth() < p.getMaxHealth() * 0.55F && p.getFoodData().getFoodLevel() < 20) {
                int slot = findFoodSlot(p);
                if (slot >= 0) { eatSlot = slot; select(p, slot); eat(mc, p, slot); return; }
            }

            if (autoWeapon) selectBestWeapon(p);

            if (escapeTicks > 0 && dist < 4.0) {
                // отступаем и держим щит вместо атаки
                backOff(mc);
                return;
            }

            if (dist <= style.reach) {
                attack(mc, p, target, dist);
            } else {
                chase(p, target, dist);
            }
            return;
        }

        // 3. цели нет — снимаем инпуты
        releaseKeys(mc);
        stuckTicks = 0;
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
        if (best != null) targetId = best.getId();
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

    // -------------------------------------------------------------- movement
    private static void chase(ClientPlayerEntity p, LivingEntity t, double dist) {
        inForward = 1.0F;
        inStrafe = 0.0F;
        inBack = false;
        inJump = shouldJump(p, t, dist);
        inSprint = dist > 3.6;
        if (stuckTicks > 12) inJump = true;
        if (Math.abs(dist - lastDist) < 0.02 && dist > 3.2) stuckTicks++; else stuckTicks = 0;
        lastDist = dist;
        p.setSprinting(inSprint);
    }

    private static boolean shouldJump(ClientPlayerEntity p, LivingEntity t, double dist) {
        if (!autoCrit) return false;
        if (!p.isOnGround()) return false;
        return dist < 3.4 && dist > 1.2 && p.getAttackStrengthScale(0.5F) < 0.6F;
    }

    private static void backOff(Minecraft mc) {
        ClientPlayerEntity p = mc.player;
        inForward = -1.0F;
        inStrafe = (p.tickCount % 10 < 5) ? 0.8F : -0.8F;
        inBack = true;
        inJump = p.isOnGround() && p.tickCount % 8 == 0;
        inSprint = false;
        p.setSprinting(false);
        raiseShield(mc, p);
    }

    private static void raiseShield(Minecraft mc, ClientPlayerEntity p) {
        ItemStack main = p.getMainHandItem();
        ItemStack off = p.getOffhandItem();
        if (off.getItem() instanceof net.minecraft.item.ShieldItem) {
            if (!p.isBlocking()) mc.gameMode.useItem(p, p.level, Hand.OFF_HAND);
        } else if (main.getItem() instanceof net.minecraft.item.ShieldItem) {
            if (!p.isBlocking()) mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
        }
    }

    // ---------------------------------------------------------------- attack
    private static void attack(Minecraft mc, ClientPlayerEntity p, LivingEntity t, double dist) {
        // W-tap: не бьём в спринте — так цель не отлетает и цепочка держится
        boolean wantWTap = dist < 3.4;
        if (wantWTap) p.setSprinting(false);
        else p.setSprinting(true);

        float charge = p.getAttackStrengthScale(0.5F);
        boolean crit = autoCrit && !p.isOnGround() && p.getDeltaMovement().y < 0.0 && !p.isSprinting();
        if (charge >= style.attackThreshold || (crit && charge >= 0.5F)) {
            if (attackingCooldown <= 0) {
                mc.gameMode.attack(p, t);
                p.swing(Hand.MAIN_HAND);
                attackingCooldown = 1;
            }
        }
        attackingCooldown = Math.max(0, attackingCooldown - 1);

        inForward = dist > 1.4 ? 0.55F : 0.0F;
        inStrafe = (p.tickCount % 20 < 10) ? 0.35F : -0.35F;
        inBack = false;
        inJump = shouldJump(p, t, dist);
        inSprint = false;
    }

    // ----------------------------------------------------------------- eat
    private static int findFoodSlot(ClientPlayerEntity p) {
        for (int i = 0; i < 9; i++) {
            ItemStack s = p.inventory.items.get(i);
            if (!s.isEmpty() && s.getItem().isEdible()) return i;
        }
        return -1;
    }

    private static void eat(Minecraft mc, ClientPlayerEntity p, int slot) {
        if (p.getHealth() >= p.getMaxHealth() * 0.92F) { eatSlot = -1; return; }
        ItemStack s = p.inventory.items.get(slot);
        if (s.isEmpty() || !s.getItem().isEdible()) { eatSlot = -1; return; }
        select(p, slot);
        if (!p.isUsingItem()) mc.gameMode.useItem(p, p.level, Hand.MAIN_HAND);
        inForward = -0.6F;
        inStrafe = 0.0F;
        inBack = true;
        inJump = false;
        inSprint = false;
        p.setSprinting(false);
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
        inForward = 0.0F;
        inStrafe = 0.0F;
        inJump = false;
        inBack = false;
        inSprint = false;
        if (p != null) p.setSprinting(false);
        eatSlot = -1;
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
        lines.add("\u00A7bAI Pilot "
                + (AiPilot.isEnabled() ? "\u00A7aON" : "\u00A7cOFF")
                + "\u00A77  [" + style.label + "]");

        if (AiPilot.isEnabled() && target != null) {
            double d = p.distanceTo(target);
            String hp = String.format(Locale.ROOT, "%.1f", target.getHealth());
            lines.add("\u00A7e  target: " + target.getName().getString() + "  " + hp + " HP  "
                    + String.format(Locale.ROOT, "%.1f", d) + "m");
            int charge = Math.round(p.getAttackStrengthScale(0.5F) * 100.0F);
            lines.add((charge > 95 ? "\u00A7a" : "\u00A77") + "  charge: " + charge + "%");
            if (escapeTicks > 0) {
                lines.add("\u00A7d  \u21B3 escape combo (backpedal + shield)");
            }
        } else if (AiPilot.isEnabled()) {
            lines.add("\u00A78  no target in " + (int) searchRange + "m");
        }
    }
}

package com.arena.aipilot;

import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.ai.attributes.AttributeModifier;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.decoration.ArmorStand;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.AxeItem;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemCooldowns;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.SwordItem;
import net.minecraft.world.item.TridentItem;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import net.minecraft.util.MovementInput;
import net.minecraft.util.text.ITextComponent;
import net.minecraft.util.text.StringTextComponent;
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
        LocalPlayer p = Minecraft.getInstance().player;
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
        LocalPlayer p = mc.player;
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
                || (target instanceof Player && (((Player) target).isCreative()
                    || ((Player) target).isSpectator()))) {
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
    private static LivingEntity findTarget(Minecraft mc, LocalPlayer p) {
        AABB box = p.getBoundingBox().inflate(searchRange, 6.0, searchRange);
        List<LivingEntity> list = mc.level.getEntitiesOfClass(LivingEntity.class, box,
                e -> e != p && e.isAlive() && e.isPickable() && !(e instanceof ArmorStand)
                        && !(e instanceof Player && (((Player) e).isCreative()
                             || ((Player) e).isSpectator() || ((Player) e).isSpectator())));
        LivingEntity best = null;
        double bestScore = Double.MAX_VALUE;
        Vec3 eye = p.position();
        for (LivingEntity e : list) {
            double d = p.distanceTo(e);
            if (d > searchRange) continue;
            // приоритет: игроки и мобы в поле зрения, ближайшие
            double score = d;
            if (!p.hasLineOfSight(e)) score += 4.0;
            if (e instanceof Player) score -= 1.5;
            if (score < bestScore) { bestScore = score; best = e; }
            if (eye.distanceToSqr(e.position()) < 0.0001) continue;
        }
        if (best != null) targetId = best.getId();
        return best;
    }

    // ------------------------------------------------------------------- aim
    private static void aimAt(LocalPlayer p, LivingEntity t, double dist) {
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
    private static void chase(LocalPlayer p, LivingEntity t, double dist) {
        MovementInput in = p.input;
        if (in == null) return;
        // идём вперёд, если цель впереди; иначе разворот делает aimAt
        in.forwardImpulse = 1.0F;
        in.leftImpulse = 0.0F;
        in.up = true;
        in.down = false;
        in.left = false;
        in.right = false;
        in.jumping = shouldJump(p, t, dist);
        p.setSprinting(dist > 3.6);
        if (stuckTicks > 12) in.jumping = true;
        if (Math.abs(dist - lastDist) < 0.02 && dist > 3.2) stuckTicks++; else stuckTicks = 0;
        lastDist = dist;
    }

    private static boolean shouldJump(LocalPlayer p, LivingEntity t, double dist) {
        if (!autoCrit) return false;
        if (!p.onGround()) return false;
        return dist < 3.4 && dist > 1.2 && p.getAttackStrengthScale(0.5F) < 0.6F;
    }

    private static void backOff(Minecraft mc) {
        LocalPlayer p = mc.player;
        MovementInput in = p.input;
        if (in == null) return;
        in.forwardImpulse = -1.0F;
        in.leftImpulse = (p.tickCount % 10 < 5) ? 0.8F : -0.8F;
        in.up = false;
        in.down = true;
        in.jumping = p.onGround() && p.tickCount % 8 == 0;
        p.setSprinting(false);
        // щит, если есть в руках
        raiseShield(mc, p);
    }

    private static void raiseShield(Minecraft mc, LocalPlayer p) {
        ItemStack main = p.getMainHandItem();
        ItemStack off = p.getOffhandItem();
        if (off.getItem() instanceof net.minecraft.world.item.ShieldItem) {
            if (!p.isBlocking()) mc.gameMode.useItem(p, InteractionHand.OFF_HAND);
        } else if (main.getItem() instanceof net.minecraft.world.item.ShieldItem) {
            if (!p.isBlocking()) mc.gameMode.useItem(p, InteractionHand.MAIN_HAND);
        }
    }

    // ---------------------------------------------------------------- attack
    private static void attack(Minecraft mc, LocalPlayer p, LivingEntity t, double dist) {
        // W-tap: не бьём в спринте — так цель не отлетает и цепочка держится
        boolean wantWTap = dist < 3.4;
        if (wantWTap) p.setSprinting(false);
        else p.setSprinting(true);

        float charge = p.getAttackStrengthScale(0.5F);
        boolean crit = autoCrit && !p.onGround() && p.getDeltaMovement().y < 0.0 && !p.isSprinting();
        if (charge >= style.attackThreshold || (crit && charge >= 0.5F)) {
            if (attackingCooldown <= 0) {
                mc.gameMode.attack(p, t);
                p.swing(InteractionHand.MAIN_HAND);
                attackingCooldown = 1;
            }
        }
        attackingCooldown = Math.max(0, attackingCooldown - 1);

        MovementInput in = p.input;
        if (in != null) {
            in.forwardImpulse = dist > 1.4 ? 0.55F : 0.0F;
            in.leftImpulse = (p.tickCount % 20 < 10) ? 0.35F : -0.35F;
            in.up = dist > 1.4;
            in.down = false;
            in.jumping = shouldJump(p, t, dist);
        }
    }

    // ----------------------------------------------------------------- eat
    private static int findFoodSlot(LocalPlayer p) {
        for (int i = 0; i < 9; i++) {
            ItemStack s = p.getInventory().items.get(i);
            if (!s.isEmpty() && s.getItem().isEdible()) return i;
        }
        return -1;
    }

    private static void eat(Minecraft mc, LocalPlayer p, int slot) {
        if (p.getHealth() >= p.getMaxHealth() * 0.92F) { eatSlot = -1; return; }
        ItemStack s = p.getInventory().items.get(slot);
        if (s.isEmpty() || !s.getItem().isEdible()) { eatSlot = -1; return; }
        select(p, slot);
        if (!p.isUsingItem()) mc.gameMode.useItem(p, InteractionHand.MAIN_HAND);
        MovementInput in = p.input;
        if (in != null) {
            in.forwardImpulse = -0.6F;
            in.leftImpulse = 0.0F;
            in.up = false; in.down = true;
            in.jumping = false;
        }
        p.setSprinting(false);
    }

    // -------------------------------------------------------------- weapons
    private static void selectBestWeapon(LocalPlayer p) {
        if (p.isUsingItem()) return;
        int best = -1;
        double bestDmg = -1.0;
        for (int i = 0; i < 9; i++) {
            ItemStack s = p.getInventory().items.get(i);
            if (s.isEmpty()) continue;
            Item it = s.getItem();
            if (!(it instanceof SwordItem || it instanceof AxeItem || it instanceof TridentItem)) continue;
            double dmg = attackDamage(s);
            if (dmg > bestDmg) { bestDmg = dmg; best = i; }
        }
        if (best >= 0 && best != p.getInventory().selected) select(p, best);
    }

    private static double attackDamage(ItemStack stack) {
        double sum = 0.0;
        for (AttributeModifier m : stack.getAttributeModifiers(EquipmentSlot.MAINHAND)
                .get(Attributes.ATTACK_DAMAGE)) {
            sum += m.getAmount();
        }
        return sum;
    }

    private static void select(LocalPlayer p, int slot) {
        p.getInventory().selected = slot;   // клиент синхронизирует слот сам
    }

    private static void releaseKeys(Minecraft mc) {
        LocalPlayer p = mc.player;
        if (p == null || p.input == null) return;
        MovementInput in = p.input;
        in.forwardImpulse = 0.0F;
        in.leftImpulse = 0.0F;
        in.up = false; in.down = false; in.left = false; in.right = false;
        in.jumping = false;
        p.setSprinting(false);
        eatSlot = -1;
    }

    /** Перехват клавиатуры, пока автобой включён (иначе игрок «сбивает» движение). */
    public static void overrideInputs(MovementInput in) {
        if (!(AiPilot.isEnabled()) || in == null) return;
        Minecraft mc = Minecraft.getInstance();
        if (mc.player == null || mc.screen != null) return;
        if (target == null) return;
        // импульсы полностью под нашим контролем
        in.leftImpulse = clamp(in.leftImpulse);
        in.forwardImpulse = clamp(in.forwardImpulse);
    }

    private static float clamp(float v) { return Math.max(-1.0F, Math.min(1.0F, v)); }

    // ------------------------------------------------------------------ HUD
    public static void renderHud(RenderGameOverlayEvent.Text event) {
        Minecraft mc = Minecraft.getInstance();
        LocalPlayer p = mc.player;
        if (p == null) return;
        List<ITextComponent> lines = event.getLeft();
        ChatFormatting stateColor = AiPilot.isEnabled() ? ChatFormatting.GREEN : ChatFormatting.RED;
        lines.add(new StringTextComponent("").append(
                new StringTextComponent("AI Pilot ").withStyle(ChatFormatting.AQUA))
                .append(new StringTextComponent(AiPilot.isEnabled() ? "ON" : "OFF").withStyle(stateColor))
                .append(new StringTextComponent("  [" + style.label + "]").withStyle(ChatFormatting.GRAY)));

        if (AiPilot.isEnabled() && target != null) {
            double d = p.distanceTo(target);
            String hp = String.format(Locale.ROOT, "%.1f", target.getHealth());
            lines.add(new StringTextComponent(
                    "  target: " + target.getName().getString() + "  " + hp + " HP  "
                            + String.format(Locale.ROOT, "%.1f", d) + "m")
                    .withStyle(ChatFormatting.YELLOW));
            int charge = Math.round(p.getAttackStrengthScale(0.5F) * 100.0F);
            lines.add(new StringTextComponent("  charge: " + charge + "%")
                    .withStyle(charge > 95 ? ChatFormatting.GREEN : ChatFormatting.GRAY));
            if (escapeTicks > 0) {
                lines.add(new StringTextComponent("  \u21B3 escape combo (backpedal + shield)")
                        .withStyle(ChatFormatting.LIGHT_PURPLE));
            }
        } else if (AiPilot.isEnabled()) {
            lines.add(new StringTextComponent("  no target in " + (int) searchRange + "m")
                    .withStyle(ChatFormatting.DARK_GRAY));
        }
    }
}

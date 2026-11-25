package com.arena.aipilot;

import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import net.minecraft.client.Minecraft;
import net.minecraft.command.CommandSource;
import net.minecraft.command.Commands;
import net.minecraft.util.text.StringTextComponent;
import net.minecraftforge.client.event.InputUpdateEvent;
import net.minecraftforge.client.event.RenderGameOverlayEvent;
import net.minecraftforge.client.event.ClientPlayerNetworkEvent;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import org.lwjgl.glfw.GLFW;

/**
 * AI Pilot — клиентский авто-бой для Minecraft 1.16.5 Forge.
 *
 * Мод не трогает сетевой протокол: он подаёт ванильные инпуты (движение, прыжок,
 * атака/использование, слоты хотбара, приседание) в клиент. Сервер видит обычного
 * игрока, поэтому работает на любом сервере и не требует ни пароля, ни сессии —
 * клиент уже запущен под вашим аккаунтом.
 *
 * Управление:
 *   O      — включить/выключить автобой
 *   R      — режим «выйти из комбо» (короткий рывок назад + щит + разворот)
 *   J      — включить/выключить HUD
 *   K      — переключить стиль боя (AGGRO / BALANCED / SAFE)
 *   .aip   — команда настроек (без слэша, как обычный чат-ввод не работает —
 *            используйте клиентскую команду "/aip ...")
 */
@Mod(AiPilot.MOD_ID)
public class AiPilot {

    public static final String MOD_ID = "aipilot";
    public static final org.apache.logging.log4j.Logger LOG =
            org.apache.logging.log4j.LogManager.getLogger(MOD_ID);

    private static boolean enabled = false;
    private static boolean hudVisible = true;
    private static boolean lastO = false;
    private static boolean lastJ = false;
    private static boolean lastK = false;

    public static boolean isEnabled() { return enabled; }
    public static boolean isHudVisible() { return hudVisible; }

    public static void toggle() {
        setEnabled(!enabled);
    }

    public static void setEnabled(boolean on) {
        enabled = on;
        AiPilotEngine.reset();
        Minecraft mc = Minecraft.getInstance();
        if (mc.player != null) {
            chat(on ? "\u00A7aAI Pilot: автобой ВКЛ" : "\u00A7cAI Pilot: автобой ВЫКЛ");
        }
    }

    public static void chat(String msg) {
        Minecraft mc = Minecraft.getInstance();
        if (mc.player != null) {
            mc.player.sendMessage(new StringTextComponent(msg), mc.player.getUUID());
        }
    }

    // ---------------------------------------------------------------- hotkeys
    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) {
        if (event.phase != TickEvent.Phase.END) return;
        Minecraft mc = Minecraft.getInstance();
        if (mc.player == null || mc.screen != null) { lastO = lastJ = lastK = false; return; }

        boolean o = GLFW.glfwGetKey(mc.getWindow().getWindow(), GLFW.GLFW_KEY_O) == GLFW.GLFW_PRESS;
        boolean j = GLFW.glfwGetKey(mc.getWindow().getWindow(), GLFW.GLFW_KEY_J) == GLFW.GLFW_PRESS;
        boolean k = GLFW.glfwGetKey(mc.getWindow().getWindow(), GLFW.GLFW_KEY_K) == GLFW.GLFW_PRESS;

        if (o && !lastO) toggle();
        if (j && !lastJ) { hudVisible = !hudVisible; }
        if (k && !lastK) {
            AiPilotEngine.cycleStyle();
        }
        lastO = o; lastJ = j; lastK = k;

        if (enabled) {
            AiPilotEngine.tick(mc);
        }
    }

    /** Жёстко перехватываем клавиатурные инпуты, пока автобой включён. */
    @SubscribeEvent
    public void onInput(InputUpdateEvent event) {
        if (!enabled) return;
        AiPilotEngine.overrideInputs(event.getMovementInput());
    }

    @SubscribeEvent
    public void onLogout(ClientPlayerNetworkEvent.LoggedOutEvent event) {
        enabled = false;
        AiPilotEngine.reset();
    }

    @SubscribeEvent
    public void onOverlay(RenderGameOverlayEvent.Text event) {
        if (!hudVisible) return;
        AiPilotEngine.renderHud(event);
    }

    // ---------------------------------------------------------------- commands
    @Mod.EventBusSubscriber(modid = MOD_ID, value = net.minecraftforge.api.distmarker.Dist.CLIENT)
    public static class Commands_ {
        @SubscribeEvent
        public static void register(net.minecraftforge.client.event.RegisterClientCommandsEvent event) {
            event.getDispatcher().register(
                Commands.literal("aip")
                    .then(Commands.literal("on").executes(c -> { setEnabled(true); return 1; }))
                    .then(Commands.literal("off").executes(c -> { setEnabled(false); return 1; }))
                    .then(Commands.literal("style")
                        .then(Commands.argument("name", StringArgumentType.word())
                            .executes(c -> {
                                AiPilotEngine.setStyle(StringArgumentType.getString(c, "name"));
                                return 1;
                            })))
                    .then(Commands.literal("range")
                        .then(Commands.argument("blocks", DoubleArgumentType.doubleArg(1.0, 6.0))
                            .executes(c -> {
                                AiPilotEngine.setRange(DoubleArgumentType.getDouble(c, "blocks"));
                                return 1;
                            })))
                    .then(Commands.literal("crit")
                        .then(Commands.argument("on", DoubleArgumentType.doubleArg(0.0, 1.0))
                            .executes(c -> {
                                AiPilotEngine.setCrit(DoubleArgumentType.getDouble(c, "on") > 0.5);
                                return 1;
                            })))
                    .executes(c -> {
                        Chat(c.getSource(), AiPilotEngine.statusLine());
                        return 1;
                    })
            );
        }

        private static void Chat(CommandSource src, String msg) {
            src.sendSuccess(new StringTextComponent(msg), false);
        }
    }
}

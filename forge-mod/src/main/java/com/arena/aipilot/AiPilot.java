package com.arena.aipilot;

import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.TextComponent;
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
 *   J      — включить/выключить HUD
 *   K      — переключить стиль боя (AGGRO / BALANCED / SAFE)
 */
@Mod(AiPilot.MOD_ID)
@Mod.EventBusSubscriber(modid = AiPilot.MOD_ID, value = net.minecraftforge.api.distmarker.Dist.CLIENT,
        bus = Mod.EventBusSubscriber.Bus.FORGE)
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
            mc.player.displayClientMessage(new TextComponent(msg), false);
        }
    }

    // ---------------------------------------------------------------- hotkeys
    @SubscribeEvent
    public static void onClientTick(TickEvent.ClientTickEvent event) {
        if (event.phase != TickEvent.Phase.START) return;
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

        if (enabled && !mc.isPaused()) {
            AiPilotEngine.tick(mc);
        }
    }

    /** Жёстко перехватываем клавиатурные инпуты, пока автобой включён. */
    @SubscribeEvent
    public static void onInput(InputUpdateEvent event) {
        if (!enabled) return;
        AiPilotEngine.overrideInputs(event.getMovementInput());
    }

    @SubscribeEvent
    public static void onLogout(ClientPlayerNetworkEvent.LoggedOutEvent event) {
        enabled = false;
        AiPilotEngine.reset();
    }

    @SubscribeEvent
    public static void onOverlay(RenderGameOverlayEvent.Text event) {
        if (!hudVisible) return;
        AiPilotEngine.renderHud(event);
    }

}

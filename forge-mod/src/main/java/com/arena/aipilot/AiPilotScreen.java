package com.arena.aipilot;

import com.mojang.blaze3d.matrix.MatrixStack;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.util.text.StringTextComponent;

/** In-game draggable sliders for the client's PvP behavior. */
public final class AiPilotScreen extends Screen {
    private static final String[] LABELS = {
            "Aggression — close distance and keep pressure",
            "Pursuit — route around walls and obstacles",
            "Strafing — lateral movement during exchanges",
            "Caution — retreat only when health is low"
    };
    private static final int TRACK_WIDTH = 280;
    private static final int ROW_HEIGHT = 38;
    private int dragging = -1;

    public AiPilotScreen() {
        super(new StringTextComponent("AI Pilot — Combat Style"));
    }

    @Override
    public void render(MatrixStack matrices, int mouseX, int mouseY, float partialTicks) {
        this.renderBackground(matrices);
        String heading = this.title.getString();
        this.font.draw(matrices, heading, (this.width - this.font.width(heading)) / 2.0F,
                this.height / 2.0F - 98.0F, 0xFFFFFFFF);

        for (int i = 0; i < LABELS.length; i++) {
            int left = this.width / 2 - TRACK_WIDTH / 2;
            int rowY = firstRowY() + i * ROW_HEIGHT;
            float value = getValue(i);
            String label = LABELS[i] + "  " + AiPilotSettings.percent(value);
            this.font.draw(matrices, label, left, rowY - 10.0F, 0xFFE8EDF2);
            int trackY = rowY + 7;
            int knobX = left + Math.round(value * TRACK_WIDTH);
            fill(matrices, left, trackY, left + TRACK_WIDTH, trackY + 4, 0xFF434A55);
            fill(matrices, left, trackY, knobX, trackY + 4, 0xFF23B7C9);
            fill(matrices, knobX - 4, trackY - 4, knobX + 5, trackY + 9, 0xFFFFFFFF);
        }

        String hint = "Drag a bar to adjust. Changes save to config/aipilot-style.properties.  [Esc] Close";
        this.font.draw(matrices, hint, (this.width - this.font.width(hint)) / 2.0F,
                firstRowY() + LABELS.length * ROW_HEIGHT + 8.0F, 0xFFB8C0CC);
        super.render(matrices, mouseX, mouseY, partialTicks);
    }

    @Override
    public boolean mouseClicked(double mouseX, double mouseY, int button) {
        if (button == 0) {
            for (int i = 0; i < LABELS.length; i++) {
                int rowY = firstRowY() + i * ROW_HEIGHT;
                if (mouseY >= rowY - 14 && mouseY <= rowY + 18
                        && mouseX >= this.width / 2.0 - TRACK_WIDTH / 2.0 - 8
                        && mouseX <= this.width / 2.0 + TRACK_WIDTH / 2.0 + 8) {
                    this.dragging = i;
                    updateFromMouse(mouseX);
                    return true;
                }
            }
        }
        return super.mouseClicked(mouseX, mouseY, button);
    }

    @Override
    public boolean mouseDragged(double mouseX, double mouseY, int button, double dragX, double dragY) {
        if (button == 0 && this.dragging >= 0) {
            updateFromMouse(mouseX);
            return true;
        }
        return super.mouseDragged(mouseX, mouseY, button, dragX, dragY);
    }

    @Override
    public boolean mouseReleased(double mouseX, double mouseY, int button) {
        if (button == 0 && this.dragging >= 0) {
            updateFromMouse(mouseX);
            this.dragging = -1;
            AiPilotSettings.save();
            return true;
        }
        return super.mouseReleased(mouseX, mouseY, button);
    }

    @Override
    public boolean keyPressed(int keyCode, int scanCode, int modifiers) {
        if (keyCode == 256) { // GLFW_KEY_ESCAPE
            AiPilotSettings.save();
            if (this.minecraft != null) this.minecraft.setScreen(null);
            return true;
        }
        return super.keyPressed(keyCode, scanCode, modifiers);
    }

    private int firstRowY() {
        return this.height / 2 - 58;
    }

    private void updateFromMouse(double mouseX) {
        double left = this.width / 2.0 - TRACK_WIDTH / 2.0;
        float value = (float) Math.max(0.0, Math.min(1.0, (mouseX - left) / TRACK_WIDTH));
        switch (this.dragging) {
            case 0: AiPilotSettings.setAggression(value); break;
            case 1: AiPilotSettings.setPursuit(value); break;
            case 2: AiPilotSettings.setStrafing(value); break;
            case 3: AiPilotSettings.setCaution(value); break;
            default: break;
        }
    }

    private static float getValue(int index) {
        switch (index) {
            case 0: return AiPilotSettings.aggression();
            case 1: return AiPilotSettings.pursuit();
            case 2: return AiPilotSettings.strafing();
            case 3: return AiPilotSettings.caution();
            default: return 0.0F;
        }
    }
}

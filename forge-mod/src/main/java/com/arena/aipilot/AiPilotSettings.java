package com.arena.aipilot;

import net.minecraftforge.fml.loading.FMLPaths;

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Properties;

/** Persistent client-side combat style controls. Values are normalized to 0..1. */
public final class AiPilotSettings {
    private static final Path FILE = FMLPaths.CONFIGDIR.get().resolve("aipilot-style.properties");
    private static final Properties PROPERTIES = new Properties();

    private static float aggression = 1.0F;
    private static float pursuit = 1.0F;
    private static float strafing = 0.65F;
    private static float caution = 0.0F;

    static {
        load();
    }

    private AiPilotSettings() {}

    public static float aggression() { return aggression; }
    public static float pursuit() { return pursuit; }
    public static float strafing() { return strafing; }
    public static float caution() { return caution; }

    public static void setAggression(float value) { aggression = clamp(value); }
    public static void setPursuit(float value) { pursuit = clamp(value); }
    public static void setStrafing(float value) { strafing = clamp(value); }
    public static void setCaution(float value) { caution = clamp(value); }

    public static String percent(float value) {
        return String.format(Locale.ROOT, "%d%%", Math.round(clamp(value) * 100.0F));
    }

    private static float clamp(float value) {
        if (Float.isNaN(value) || Float.isInfinite(value)) return 0.0F;
        return Math.max(0.0F, Math.min(1.0F, value));
    }

    private static void load() {
        if (!Files.isRegularFile(FILE)) return;
        try (InputStream input = Files.newInputStream(FILE)) {
            PROPERTIES.load(input);
            aggression = read("aggression", aggression);
            pursuit = read("pursuit", pursuit);
            strafing = read("strafing", strafing);
            caution = read("caution", caution);
        } catch (Exception error) {
            AiPilot.LOG.warn("Could not read AI Pilot style settings; using defaults", error);
        }
    }

    private static float read(String key, float fallback) {
        try {
            return clamp(Float.parseFloat(PROPERTIES.getProperty(key, Float.toString(fallback))));
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    public static void save() {
        PROPERTIES.setProperty("aggression", Float.toString(aggression));
        PROPERTIES.setProperty("pursuit", Float.toString(pursuit));
        PROPERTIES.setProperty("strafing", Float.toString(strafing));
        PROPERTIES.setProperty("caution", Float.toString(caution));
        try {
            Files.createDirectories(FILE.getParent());
            try (OutputStream output = Files.newOutputStream(FILE)) {
                PROPERTIES.store(output, "AI Pilot combat style (normalized values 0..1)");
            }
        } catch (Exception error) {
            AiPilot.LOG.warn("Could not save AI Pilot style settings", error);
        }
    }
}

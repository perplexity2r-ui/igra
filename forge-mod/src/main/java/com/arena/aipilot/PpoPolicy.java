package com.arena.aipilot;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Pure-Java deterministic inference for the multi-opponent SB3 PPO actor.
 * The binary resource contains the 50-feature input, two 192-wide tanh layers, and 30 action logits.
 */
public final class PpoPolicy {
    private static final String RESOURCE = "/com/arena/aipilot/ppo_policy.bin";
    private static final int OBS = 50;
    private static final int HIDDEN_1 = 192;
    private static final int HIDDEN_2 = 192;
    private static final int ACTIONS = 30;
    private static final int HEADER_BYTES = 24;
    private static final int EXPECTED_BYTES = HEADER_BYTES
            + (HIDDEN_1 * OBS + HIDDEN_1
            + HIDDEN_2 * HIDDEN_1 + HIDDEN_2
            + ACTIONS * HIDDEN_2 + ACTIONS) * 4;

    private final float[] w1 = new float[HIDDEN_1 * OBS];
    private final float[] b1 = new float[HIDDEN_1];
    private final float[] w2 = new float[HIDDEN_2 * HIDDEN_1];
    private final float[] b2 = new float[HIDDEN_2];
    private final float[] w3 = new float[ACTIONS * HIDDEN_2];
    private final float[] b3 = new float[ACTIONS];
    private final float[] h1 = new float[HIDDEN_1];
    private final float[] h2 = new float[HIDDEN_2];
    private final float[] logits = new float[ACTIONS];
    private boolean ready;

    public PpoPolicy() {
        try {
            byte[] bytes = readResource();
            if (bytes.length != EXPECTED_BYTES) {
                throw new IOException("Unexpected PPO weight file length: " + bytes.length);
            }
            ByteBuffer data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            if (data.get() != 'A' || data.get() != 'I' || data.get() != 'P' || data.get() != 'P') {
                throw new IOException("Invalid PPO weight file signature");
            }
            int version = data.getInt();
            int obs = data.getInt();
            int hidden1 = data.getInt();
            int hidden2 = data.getInt();
            int actions = data.getInt();
            if (version != 1 || obs != OBS || hidden1 != HIDDEN_1
                    || hidden2 != HIDDEN_2 || actions != ACTIONS) {
                throw new IOException("Unsupported PPO actor shape/version");
            }
            readFloats(data, w1);
            readFloats(data, b1);
            readFloats(data, w2);
            readFloats(data, b2);
            readFloats(data, w3);
            readFloats(data, b3);
            ready = true;
            AiPilot.LOG.info("Loaded PPO multi-opponent actor: {} -> {} -> {} -> {}",
                    OBS, HIDDEN_1, HIDDEN_2, ACTIONS);
        } catch (Exception e) {
            ready = false;
            AiPilot.LOG.error("Could not load multi-opponent PPO actor; auto-combat is disabled", e);
        }
    }

    public boolean isReady() {
        return ready;
    }

    /** Returns the deterministic categorical action (argmax over the actor's logits). */
    public int predict(float[] observation) {
        if (!ready || observation == null || observation.length != OBS) return 0;
        for (int j = 0; j < HIDDEN_1; j++) {
            float sum = b1[j];
            int row = j * OBS;
            for (int i = 0; i < OBS; i++) sum += w1[row + i] * observation[i];
            h1[j] = (float) Math.tanh(sum);
        }
        for (int j = 0; j < HIDDEN_2; j++) {
            float sum = b2[j];
            int row = j * HIDDEN_1;
            for (int i = 0; i < HIDDEN_1; i++) sum += w2[row + i] * h1[i];
            h2[j] = (float) Math.tanh(sum);
        }
        for (int a = 0; a < ACTIONS; a++) {
            float sum = b3[a];
            int row = a * HIDDEN_2;
            for (int i = 0; i < HIDDEN_2; i++) sum += w3[row + i] * h2[i];
            logits[a] = sum;
        }
        int best = 0;
        float bestLogit = -Float.MAX_VALUE;
        for (int a = 0; a < ACTIONS; a++) {
            if (logits[a] > bestLogit) {
                bestLogit = logits[a];
                best = a;
            }
        }
        return best;
    }

    private static byte[] readResource() throws IOException {
        InputStream stream = PpoPolicy.class.getResourceAsStream(RESOURCE);
        if (stream == null) throw new IOException("Missing resource " + RESOURCE);
        try (InputStream in = stream; ByteArrayOutputStream out = new ByteArrayOutputStream(EXPECTED_BYTES)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = in.read(buffer)) >= 0) {
                if (count > 0) out.write(buffer, 0, count);
            }
            return out.toByteArray();
        }
    }

    private static void readFloats(ByteBuffer source, float[] destination) throws IOException {
        if (source.remaining() < destination.length * 4) throw new IOException("Truncated PPO weights");
        for (int i = 0; i < destination.length; i++) destination[i] = source.getFloat();
    }
}

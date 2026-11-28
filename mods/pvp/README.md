# AI Pilot 7 — PvP combat mod

**Displayed in Minecraft:** `AI Pilot 7 — авто-бой`

**Mod version:** `1.0.7`
**Artifact:** [`aipilot-1.16.5.jar`](aipilot-1.16.5.jar)

- Minecraft Java **1.16.5**, Forge **36.2.39**.
- Copy the JAR into the `mods` folder of the Forge profile you launch.
- Press **O** to toggle AI combat, **J** to toggle the HUD, and **K** to open combat-style sliders for aggression, pursuit, strafing, and caution. Slider values save to `config/aipilot-style.properties`.
- The client mod uses the currently logged-in Minecraft session and ordinary client actions; it does not ask for account credentials or log in separately.

The GitHub Actions build publishes the Forge JAR to this folder on the source branch. The packaged actor includes the bow/crossbow fine-tuning; its deterministic simulator evaluation confirmed shots and hits from both weapons. No live Minecraft server test has been run.

Source and integration notes are in [`../../forge-mod/`](../../forge-mod/) and [`../../forge-mod/PPO_INTEGRATION.md`](../../forge-mod/PPO_INTEGRATION.md). Use automation only where permitted; public servers may prohibit bots or use anti-cheat.

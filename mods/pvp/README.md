# AI Pilot — PvP combat mod

**Artifact:** [`aipilot-1.16.5.jar`](aipilot-1.16.5.jar)

- Minecraft Java **1.16.5**, Forge **36.2.39**.
- Copy the JAR into the `mods` folder of the Forge profile you launch.
- In the updated source, press **O** to toggle AI combat, **J** to toggle the HUD, and **K** to open combat-style sliders for aggression, pursuit, strafing, and caution. Slider values save to `config/aipilot-style.properties`.
- The client mod uses the currently logged-in Minecraft session and ordinary client actions; it does not ask for account credentials or log in separately.

**Build status:** the checked-in JAR is still the previous build. The Forge sources now expect the terrain-aware 44-feature actor and include the new navigation and style controls; rebuild with the Minecraft 1.16.5 / Forge 36.2.39 toolchain before installing these source changes.

The PPO actor is embedded in the JAR. Source and integration notes are in [`../../forge-mod/`](../../forge-mod/) and [`../../forge-mod/PPO_INTEGRATION.md`](../../forge-mod/PPO_INTEGRATION.md). Use automation only where permitted; public servers may prohibit bots or use anti-cheat.

// Full-Physics Minecraft 1.16.5 PvP Arena Server (Node.js / Prismarine Protocol)
// Supports real Mineflayer 1.16.5 clients (`LearnerBot` & `EnemyBot`) with:
//   - Real chunk world + real-time `/setblock` & `/fill` (`block_change` packets so Mineflayer
//     `prismarine-physics` collides with placed cobblestone/obsidian/plank/hay barricades,
//     slows down inside cobwebs, climbs pillar blocks, and flows in water!)
//   - Full `/give`, `/clear`, `/effect give|clear` command engine (12 weapons/tools with
//     Knockback/Sharpness/Fire Aspect, 5 armor tiers, shields, totems of undying, golden
//     apples, splash healing, ender pearls, strength+speed buffs, snowballs/eggs/crossbows/harming)
//   - 1.16.5 combat cooldowns, sprinting knockback, critical hits, shield blocking & axe
//     5-second shield disable, fire/lava burn ticks, regeneration, absorption, and Totem pops!
//   - Built-in HTTP + Live 3D 8-Camera 360° Spectator Web Dashboard on port 8080!

const http = require('http')
const mc = require('minecraft-protocol')
const { WebSocketServer } = require('ws')
const Chunk = require('prismarine-chunk')('1.16.5')
const { Vec3 } = require('vec3')
const mcData = require('minecraft-data')('1.16.5')

const WEAPON_STATS = {
  stick: { dmg: 1.0, cdMs: 250, kb: 1.0, axe: false },
  wooden_shovel: { dmg: 2.5, cdMs: 1000, kb: 1.0, axe: false },
  golden_hoe: { dmg: 1.0, cdMs: 250, kb: 1.0, axe: false },
  stone_sword: { dmg: 5.0, cdMs: 625, kb: 1.0, axe: false },
  iron_pickaxe: { dmg: 4.0, cdMs: 830, kb: 1.0, axe: false },
  iron_sword: { dmg: 6.0, cdMs: 625, kb: 1.0, axe: false },
  iron_axe: { dmg: 9.0, cdMs: 1100, kb: 1.1, axe: true },
  diamond_sword: { dmg: 7.0, cdMs: 625, kb: 1.0, axe: false },
  trident: { dmg: 9.0, cdMs: 910, kb: 1.1, axe: false },
  diamond_axe: { dmg: 9.0, cdMs: 1000, kb: 1.1, axe: true },
  netherite_sword: { dmg: 8.0, cdMs: 625, kb: 1.2, axe: false },
  netherite_axe: { dmg: 10.0, cdMs: 1000, kb: 1.2, axe: true },
}

const ARMOR_POINTS = {
  leather_chestplate: 3,
  leather_boots: 1,
  golden_helmet: 2,
  chainmail_chestplate: 5,
  chainmail_leggings: 4,
  iron_helmet: 2,
  iron_chestplate: 6,
  iron_leggings: 5,
  iron_boots: 2,
  diamond_helmet: 3,
  diamond_chestplate: 8,
  diamond_boots: 3,
}

function getBlockStateId(blockName) {
  const clean = blockName.replace(/^minecraft:/, '').trim()
  const blk = mcData.blocksByName[clean]
  return blk ? blk.defaultState : 0
}

function startPvPServer({
  mcPort = Number(process.env.MC_PORT || 25565),
  mcHost = process.env.MC_HOST || '127.0.0.1',
  webPort = Number(process.env.WEB_PORT || 8080),
} = {}) {
  // Build 4 chunks (-1..0, -1..0) covering world X,Z in [-16..15]
  const chunks = new Map()
  const keyChunk = (cx, cz) => `${cx},${cz}`
  for (const cx of [-1, 0]) {
    for (const cz of [-1, 0]) {
      const ch = new Chunk()
      for (let lx = 0; lx < 16; lx++) {
        for (let lz = 0; lz < 16; lz++) {
          const wx = cx * 16 + lx
          const wz = cz * 16 + lz
          if (wx >= -10 && wx <= 10 && wz >= -10 && wz <= 10) {
            ch.setBlockStateId(new Vec3(lx, 64, lz), getBlockStateId('stone'))
            if (wx === -10 || wx === 10 || wz === -10 || wz === 10) {
              for (let y = 65; y <= 72; y++) {
                ch.setBlockStateId(new Vec3(lx, y, lz), getBlockStateId('bedrock'))
              }
            }
          } else {
            ch.setBlockStateId(new Vec3(lx, 64, lz), getBlockStateId('grass_block'))
          }
        }
      }
      chunks.set(keyChunk(cx, cz), ch)
    }
  }

  // Track dynamic blocks placed inside arena: "x,y,z" -> { x, y, z, name, stateId }
  const arenaBlocks = new Map()
  const combatEvents = []
  const pushEvent = (txt) => {
    const line = `[${new Date().toISOString().slice(11, 19)}] ${txt}`
    combatEvents.push(line)
    if (combatEvents.length > 40) combatEvents.shift()
    console.log(line)
  }

  const players = new Map() // entityId -> playerState

  function broadcastAll(pkt, data) {
    for (const p of players.values()) {
      try { p.client.write(pkt, data) } catch {}
    }
  }

  function broadcastExcept(senderId, pkt, data) {
    for (const [id, p] of players.entries()) {
      if (id !== senderId) {
        try { p.client.write(pkt, data) } catch {}
      }
    }
  }

  function setWorldBlock(x, y, z, blockName) {
    const clean = blockName.replace(/^minecraft:/, '').split('[')[0].trim()
    const stateId = getBlockStateId(clean)
    const cx = Math.floor(x / 16)
    const cz = Math.floor(z / 16)
    const ch = chunks.get(keyChunk(cx, cz))
    if (ch && y >= 0 && y < 256) {
      const lx = ((x % 16) + 16) % 16
      const lz = ((z % 16) + 16) % 16
      ch.setBlockStateId(new Vec3(lx, y, lz), stateId)
    }
    const bKey = `${x},${y},${z}`
    if (clean === 'air') {
      arenaBlocks.delete(bKey)
    } else if (x >= -9 && x <= 9 && z >= -9 && z <= 9 && y >= 65 && y <= 72) {
      arenaBlocks.set(bKey, { x, y, z, name: clean, stateId })
    }
    broadcastAll('block_change', {
      location: { x, y, z },
      type: stateId,
    })
  }

  function syncHealth(p) {
    try {
      p.client.write('update_health', {
        health: Math.max(0.5, Math.min(20, p.hp)),
        food: 20,
        foodSaturation: 5.0,
      })
    } catch {}
  }

  function applyDamage(target, rawDmg, opts = {}) {
    if (!target || rawDmg <= 0) return 0
    const reduction = opts.bypassArmor ? 0 : Math.min(0.76, target.armorPoints * 0.038)
    const mitigated = rawDmg * (1.0 - reduction)
    const usedAbs = Math.min(target.absorption, mitigated)
    const rem = mitigated - usedAbs
    target.absorption = Math.max(0, target.absorption - usedAbs)
    let nextHp = Math.max(0.5, target.hp - rem)

    // Totem of Undying resurrection check!
    if (nextHp <= 6.0 && target.hasTotem) {
      target.hasTotem = false
      nextHp = 12.0
      target.absorption = 4.0
      target.regenUntil = Date.now() + 4500
      target.fireUntil = 0
      pushEvent(`⚡ ${target.username} POPPED TOTEM OF UNDYING! (+HP & Absorption)`)
    }
    const lost = Math.max(0, target.hp - nextHp) + usedAbs
    target.hp = nextHp
    syncHealth(target)
    return lost
  }

  function findPlayerByName(name) {
    for (const p of players.values()) {
      if (p.username === name) return p
    }
    return null
  }

  function parseGiveItem(rawSpec, count) {
    const clean = rawSpec.replace(/^minecraft:/, '')
    const braceIdx = clean.indexOf('{')
    const itemName = (braceIdx >= 0 ? clean.slice(0, braceIdx) : clean).trim()
    const nbtRaw = braceIdx >= 0 ? clean.slice(braceIdx) : ''
    const itemDef = mcData.itemsByName[itemName]
    if (!itemDef) return null
    return {
      itemName,
      itemId: itemDef.id,
      count: Math.max(1, Number(count) || 1),
      sharpness: nbtRaw.includes('sharpness') ? (nbtRaw.includes('lvl:3') ? 3 : 2) : 0,
      knockback: nbtRaw.includes('knockback') ? 2 : 0,
      fireAspect: nbtRaw.includes('fire_aspect') ? (nbtRaw.includes('lvl:2') ? 2 : 1) : 0,
      impaling: nbtRaw.includes('impaling') ? 2 : 0,
      potion: nbtRaw.includes('harming') ? 'harming' : (nbtRaw.includes('healing') ? 'healing' : ''),
    }
  }

  function handleCommand(sender, rawMsg) {
    const parts = rawMsg.trim().split(/\s+/)
    const cmd = parts[0]
    if (cmd === '/fill' && parts.length >= 8) {
      const x1 = Number(parts[1]), y1 = Number(parts[2]), z1 = Number(parts[3])
      const x2 = Number(parts[4]), y2 = Number(parts[5]), z2 = Number(parts[6])
      const blk = parts[7]
      const lx = Math.min(x1, x2), hx = Math.max(x1, x2)
      const ly = Math.min(y1, y2), hy = Math.max(y1, y2)
      const lz = Math.min(z1, z2), hz = Math.max(z1, z2)
      // Fast reset when clearing interior arena air
      if (blk.includes('air') && lx <= -9 && hx >= 9 && ly === 65) {
        for (const b of Array.from(arenaBlocks.values())) {
          setWorldBlock(b.x, b.y, b.z, 'air')
        }
        // Reset bot positions cleanly on episode reset!
        for (const p of players.values()) {
          p.x = p.username === 'LearnerBot' ? -3.5 : 3.5
          p.y = 65.0
          p.z = p.username === 'LearnerBot' ? -2.5 : 2.5
          try {
            p.client.write('position', {
              x: p.x, y: p.y, z: p.z, yaw: p.yaw, pitch: p.pitch, flags: 0x00, teleportId: Math.floor(Math.random() * 9000) + 10,
            })
            broadcastExcept(p.id, 'entity_teleport', {
              entityId: p.id, x: p.x, y: p.y, z: p.z, yaw: 0, pitch: 0, onGround: true,
            })
          } catch {}
        }
        return
      }
      if ((hx - lx + 1) * (hy - ly + 1) * (hz - lz + 1) <= 600) {
        for (let x = lx; x <= hx; x++) {
          for (let y = ly; y <= hy; y++) {
            for (let z = lz; z <= hz; z++) {
              setWorldBlock(x, y, z, blk)
            }
          }
        }
      }
    } else if (cmd === '/setblock' && parts.length >= 5) {
      const x = Number(parts[1]), y = Number(parts[2]), z = Number(parts[3])
      const blk = parts[4]
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        setWorldBlock(x, y, z, blk)
        const clean = blk.replace(/^minecraft:/, '')
        if (clean !== 'air') {
          pushEvent(`🧱 Block placed at (${x}, ${y}, ${z}): ${clean}`)
        }
      }
    } else if (cmd === '/clear' && parts[1]) {
      const p = findPlayerByName(parts[1])
      if (p) {
        p.nextSlot = 36
        p.inventory.clear()
        p.weapon = { itemName: 'fist', dmg: 1.0, cdMs: 250, kb: 1.0, axe: false, fire: 0 }
        p.armorPoints = 0
        p.armorPieces = []
        p.hasShield = false
        p.hasTotem = false
        p.shieldUp = false
        p.crystals = 0
        p.anchors = 0
        p.minecarts = 0
        p.obsidian = 0
        p.cagedUntil = 0
        for (let s = 5; s <= 45; s++) {
          try {
            p.client.write('set_slot', { windowId: 0, slot: s, item: { present: false } })
          } catch {}
        }
      }
    } else if (cmd === '/effect' && parts[1] === 'clear' && parts[2]) {
      const p = findPlayerByName(parts[2])
      if (p) {
        p.absorption = 0
        p.regenUntil = 0
        p.speedUntil = 0
        p.strengthUntil = 0
        p.fireUntil = 0
        p.shieldDisabledUntil = 0
      }
    } else if (cmd === '/effect' && parts[1] === 'give' && parts[2] && parts[3]) {
      const p = findPlayerByName(parts[2])
      if (!p) return
      const eff = parts[3].replace(/^minecraft:/, '')
      const durSec = Number(parts[4] || 5)
      const amp = Number(parts[5] || 0)
      const now = Date.now()
      if (eff === 'instant_health') {
        if (amp >= 5) p.hp = 20
        else p.hp = Math.min(20, p.hp + (amp >= 1 ? 8 : 4))
        syncHealth(p)
      } else if (eff === 'regeneration') {
        p.regenUntil = now + durSec * 1000
      } else if (eff === 'absorption') {
        p.absorption = 4.0
      } else if (eff === 'speed') {
        p.speedUntil = now + durSec * 1000
      } else if (eff === 'strength') {
        p.strengthUntil = now + durSec * 1000
      }
    } else if (cmd === '/arena' && parts[1] === 'explode' && parts.length >= 8) {
      // Explosion engine for crystals / anchors / TNT carts (wired to bot.js techniques)
      const ex = Number(parts[2]), ey = Number(parts[3]), ez = Number(parts[4])
      const power = Number(parts[5]) || 10
      const selfPower = Number(parts[6]) || 16
      const casterName = parts[7]
      const caster = findPlayerByName(casterName)
      for (const p of players.values()) {
        if (p.id === caster?.id) continue
        const d = Math.hypot(p.x - ex, p.z - ez) + Math.abs(p.y - ey) * 0.6
        const raw = Math.max(0, power - d * 2.4)
        // Obsidian bunker / shield absorbs blast damage
        const cageMul = Date.now() < (p.cagedUntil || 0) ? 0.35 : 1.0
        const shieldMul = p.shieldUp && Date.now() >= p.shieldDisabledUntil ? 0.7 : 1.0
        const dist = Math.hypot(p.x - ex, p.z - ez)
        if (raw <= 0.4) {
          pushEvent(`💥 Blast missed ${p.username} (out of ${power.toFixed(0)}-power radius, dist ${dist.toFixed(1)})`)
        } else if (raw * cageMul * shieldMul <= 0.4) {
          pushEvent(`🛡️ ${p.username} absorbed the blast (cage/shield)`)
        }
        if (raw * cageMul * shieldMul > 0.4) {
          applyDamage(p, raw * cageMul * shieldMul, { blast: true })
          try {
            const len = Math.hypot(p.x - ex, p.z - ez) || 1
            p.client.write('entity_velocity', {
              entityId: p.id,
              velocity: {
                x: Math.round(((p.x - ex) / len) * 5200),
                y: 2400,
                z: Math.round(((p.z - ez) / len) * 5200),
              },
            })
          } catch {}
        }
        pushEvent(`💥 Explosion at (${ex.toFixed(1)}, ${ez.toFixed(1)}) hit ${p.username} (-${(raw * cageMul * shieldMul).toFixed(1)})`)
      }
      if (caster) {
        const d = Math.hypot(caster.x - ex, caster.z - ez)
        const selfRaw = Math.max(0, selfPower - d * 2.6)
        const cageMul = Date.now() < (caster.cagedUntil || 0) ? 0.35 : 1.0
        if (selfRaw * cageMul > 0.4) {
          applyDamage(caster, selfRaw * cageMul, { blast: true })
          pushEvent(`💥 ${caster.username} caught own blast (-${(selfRaw * cageMul).toFixed(1)} HP)`)
        }
      }
    } else if (cmd === '/arena' && parts[1] === 'cage' && parts[2]) {
      // 4-block obsidian blast cage around the bot (temporary)
      const p = findPlayerByName(parts[2])
      if (p) {
        p.cagedUntil = Date.now() + (Number(parts[3]) || 4000)
        for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          setWorldBlock(Math.round(p.x) + dx, 65, Math.round(p.z) + dz, 'minecraft:obsidian')
          setWorldBlock(Math.round(p.x) + dx, 66, Math.round(p.z) + dz, 'minecraft:obsidian')
        }
        pushEvent(`🧱 ${p.username} built an OBSIDIAN BLAST CAGE`)
        setTimeout(() => {
          for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            try {
              setWorldBlock(Math.round(p.x) + dx, 65, Math.round(p.z) + dz, 'minecraft:air')
              setWorldBlock(Math.round(p.x) + dx, 66, Math.round(p.z) + dz, 'minecraft:air')
            } catch {}
          }
        }, Number(parts[3]) || 4000)
      }
    } else if (cmd === '/arena' && parts[1] === 'wtap' && parts[2]) {
      const p = findPlayerByName(parts[2])
      if (p) pushEvent(`⚡ ${p.username} executed a W-TAP sprint-reset combo`)
    } else if (cmd === '/arena' && parts[1] === 'combobreak' && parts[2]) {
      const p = findPlayerByName(parts[2])
      if (p) pushEvent(`🛡️ ${p.username} BROKE OUT of a combo chain`)
    } else if (cmd === '/arena' && parts[1] === 'counter' && parts[2]) {
      const p = findPlayerByName(parts[2])
      if (p) pushEvent(`⚔️ ${p.username} landed a COMBO COUNTER`)
    } else if (cmd === '/arena' && parts[1] === 'blink' && parts[3] && parts[4]) {
      const p = findPlayerByName(parts[2])
      if (p) {
        p.x = Number(parts[3]) || p.x
        p.z = Number(parts[4]) || p.z
        p.velocity = { x: 0, y: 0, z: 0 }
        try { p.client.write('position', { x: p.x, y: p.y, z: p.z, yaw: 0, pitch: 0, flags: 0, teleportId: 1 }) } catch {}
        pushEvent(`🔮 ${p.username} pearl-blinked away to escape`)
      }
    } else if (cmd === '/give' && parts[1] && parts[2]) {
      const p = findPlayerByName(parts[1])
      if (!p) return
      const parsed = parseGiveItem(parts[2], parts[3] || 1)
      if (!parsed) return
      const slot = p.nextSlot <= 44 ? p.nextSlot++ : 36
      p.inventory.set(slot, parsed)
      try {
        p.client.write('set_slot', {
          windowId: 0,
          slot,
          item: { present: true, itemId: parsed.itemId, itemCount: parsed.count },
        })
      } catch {}

      if (WEAPON_STATS[parsed.itemName]) {
        const base = WEAPON_STATS[parsed.itemName]
        const sharpBonus = parsed.sharpness > 0 ? 0.5 * parsed.sharpness + 0.5 : 0
        p.weapon = {
          itemName: parsed.itemName,
          dmg: base.dmg + sharpBonus,
          cdMs: base.cdMs,
          kb: base.kb + parsed.knockback * 0.65,
          axe: base.axe,
          fire: parsed.fireAspect * 3000,
        }
      } else if (ARMOR_POINTS[parsed.itemName]) {
        p.armorPoints += ARMOR_POINTS[parsed.itemName]
        p.armorPieces.push(parsed.itemName)
      } else if (parsed.itemName === 'shield') {
        p.hasShield = true
      } else if (parsed.itemName === 'totem_of_undying') {
        p.hasTotem = true
      } else if (parsed.itemName === 'end_crystal') {
        p.crystals = (p.crystals || 0) + parsed.count
        pushEvent(`[give] ${p.username} +${parsed.count} end_crystal (now ${p.crystals})`)
      } else if (parsed.itemName === 'respawn_anchor') {
        p.anchors = (p.anchors || 0) + parsed.count
        pushEvent(`[give] ${p.username} +${parsed.count} respawn_anchor (now ${p.anchors})`)
      } else if (parsed.itemName === 'tnt_minecart') {
        p.minecarts = (p.minecarts || 0) + parsed.count
        pushEvent(`[give] ${p.username} +${parsed.count} tnt_minecart (now ${p.minecarts})`)
      } else if (parsed.itemName === 'obsidian') {
        p.obsidian = (p.obsidian || 0) + parsed.count
        pushEvent(`[give] ${p.username} +${parsed.count} obsidian (now ${p.obsidian})`)
      }
    }
  }

  const server = mc.createServer({
    'online-mode': false,
    host: mcHost,
    port: mcPort,
    version: '1.16.5',
  })

  function uuidFor(name) {
    if (name === 'LearnerBot') return '23b95b27-c1ca-34d4-814c-c031f966a195'
    if (name === 'EnemyBot') return '461da519-6747-3593-ac2c-ddf98075b053'
    return '00000000-0000-3000-8000-' + String(Date.now()).slice(-12).padStart(12, '0')
  }

  server.on('login', (client) => {
    const entityId = client.id + 100
    const isLearner = client.username === 'LearnerBot'
    const state = {
      id: entityId,
      username: client.username,
      uuid: uuidFor(client.username),
      client,
      x: isLearner ? -3.5 : 3.5,
      y: 65.0,
      z: isLearner ? -2.5 : 2.5,
      yaw: isLearner ? 0 : 180,
      pitch: 0,
      onGround: true,
      sprinting: false,
      hp: 20,
      absorption: 0,
      lastHurt: 0,
      lastAttack: 0,
      nextSlot: 36,
      heldSlot: 36,
      inventory: new Map(),
      weapon: { itemName: 'diamond_sword', dmg: 7.0, cdMs: 625, kb: 1.0, axe: false, fire: 0 },
      armorPoints: 0,
      armorPieces: [],
      hasShield: false,
      hasTotem: false,
      shieldUp: false,
      shieldDisabledUntil: 0,
      regenUntil: 0,
      speedUntil: 0,
      strengthUntil: 0,
      fireUntil: 0,
      swingUntil: 0,
      crystals: 0,
      anchors: 0,
      minecarts: 0,
      obsidian: 0,
      cagedUntil: 0,
      /** last technique the player executed (surfaced to Python / spectator) */
      lastTech: null,
      techLog: [],
    }
    players.set(entityId, state)
    pushEvent(`✅ ${client.username} joined live 1.16.5 arena at (${state.x}, ${state.y}, ${state.z})`)

    client.write('login', {
      entityId,
      isHardcore: false,
      gameMode: 0,
      previousGameMode: 255,
      worldNames: mcData.loginPacket.worldNames,
      dimensionCodec: mcData.loginPacket.dimensionCodec,
      dimension: mcData.loginPacket.dimension,
      worldName: 'minecraft:overworld',
      hashedSeed: [0, 0],
      maxPlayers: 10,
      viewDistance: 10,
      reducedDebugInfo: false,
      enableRespawnScreen: true,
      isDebug: false,
      isFlat: true,
    })

    for (const cx of [-1, 0]) {
      for (const cz of [-1, 0]) {
        const ch = chunks.get(keyChunk(cx, cz))
        client.write('map_chunk', {
          x: cx,
          z: cz,
          groundUp: true,
          bitMap: ch.getMask(),
          heightmaps: { type: 'compound', name: '', value: {} },
          biomes: ch.dumpBiomes(),
          chunkData: ch.dump(),
          blockEntities: [],
        })
      }
    }

    client.write('position', {
      x: state.x, y: state.y, z: state.z, yaw: state.yaw, pitch: state.pitch, flags: 0x00, teleportId: 1,
    })
    syncHealth(state)

    for (const [otherId, other] of players.entries()) {
      client.write('player_info', {
        action: 'add_player',
        data: [{ uuid: other.uuid, name: other.username, properties: [], gamemode: 0, ping: 1, displayName: undefined }],
      })
      if (otherId !== entityId) {
        client.write('named_entity_spawn', {
          entityId: other.id, playerUUID: other.uuid, x: other.x, y: other.y, z: other.z, yaw: 0, pitch: 0, metadata: [],
        })
        other.client.write('player_info', {
          action: 'add_player',
          data: [{ uuid: state.uuid, name: state.username, properties: [], gamemode: 0, ping: 1, displayName: undefined }],
        })
        other.client.write('named_entity_spawn', {
          entityId: state.id, playerUUID: state.uuid, x: state.x, y: state.y, z: state.z, yaw: 0, pitch: 0, metadata: [],
        })
      }
    }

    const checkFinite = (pktName, data) => {
      for (const [k, v] of Object.entries(data)) {
        if (typeof v === 'number' && !Number.isFinite(v)) {
          client.end('Invalid move player packet received')
          throw new Error(`Invalid move packet from ${client.username}: ${pktName}.${k}=${v}`)
        }
      }
    }

    const syncPos = () => {
      // Clamp inside bedrock walls [-9.4, 9.4]
      state.x = Math.max(-9.4, Math.min(9.4, state.x))
      state.z = Math.max(-9.4, Math.min(9.4, state.z))
      broadcastExcept(entityId, 'entity_teleport', {
        entityId, x: state.x, y: state.y, z: state.z, yaw: 0, pitch: 0, onGround: state.onGround,
      })
    }

    client.on('position', (d) => {
      checkFinite('position', d)
      state.x = d.x; state.y = d.y; state.z = d.z; state.onGround = Boolean(d.onGround)
      syncPos()
    })

    client.on('position_look', (d) => {
      checkFinite('position_look', d)
      state.x = d.x; state.y = d.y; state.z = d.z; state.yaw = d.yaw; state.pitch = d.pitch; state.onGround = Boolean(d.onGround)
      syncPos()
    })

    client.on('look', (d) => {
      checkFinite('look', d)
      state.yaw = d.yaw; state.pitch = d.pitch
    })

    client.on('entity_action', (d) => {
      if (d.actionId === 3) state.sprinting = true
      else if (d.actionId === 4) state.sprinting = false
    })

    client.on('held_item_slot', (d) => {
      state.heldSlot = 36 + Number(d.slotId || 0)
    })

    client.on('arm_animation', () => {
      state.swingUntil = Date.now() + 260
    })

    client.on('use_item', (d) => {
      const now = Date.now()
      // Off-hand shield block (hand === 1)
      if (d.hand === 1 && state.hasShield && now >= state.shieldDisabledUntil) {
        state.shieldUp = true
        return
      }
      // Check held item for Ender Pearl / Snowball / Egg / Crossbow / Harming Splash
      const held = state.inventory.get(state.heldSlot)
      const peer = Array.from(players.values()).find((p) => p.id !== state.id)
      if (!held || !peer) return

      if (held.itemName === 'ender_pearl') {
        const dx = peer.x - state.x
        const dz = peer.z - state.z
        const dist = Math.hypot(dx, dz) || 1
        const nx = state.hp <= 11.5
          ? Math.max(-8.8, Math.min(8.8, peer.x - (dx / dist) * 7.5))
          : Math.max(-8.8, Math.min(8.8, peer.x - (dx / dist) * 2.0))
        const nz = state.hp <= 11.5
          ? Math.max(-8.8, Math.min(8.8, peer.z - (dz / dist) * 7.5))
          : Math.max(-8.8, Math.min(8.8, peer.z - (dz / dist) * 2.0))
        state.x = nx; state.z = nz; state.y = 65.0
        applyDamage(state, 1.5, { bypassArmor: true })
        try {
          state.client.write('position', { x: state.x, y: state.y, z: state.z, yaw: state.yaw, pitch: state.pitch, flags: 0x00, teleportId: Math.floor(Math.random() * 9000) + 10 })
        } catch {}
        syncPos()
        pushEvent(`🟣 ${state.username} blinked via Ender Pearl to (${nx.toFixed(1)}, ${nz.toFixed(1)})`)
      } else if (['snowball', 'egg', 'crossbow', 'splash_potion'].includes(held.itemName)) {
        const dx = peer.x - state.x
        const dz = peer.z - state.z
        const dist = Math.hypot(dx, dz) || 1
        if (dist <= 16.0) {
          const rDmg = held.itemName === 'crossbow' ? 7.0 : (held.potion === 'harming' ? 6.0 : 0.8)
          if (peer.shieldUp && now >= peer.shieldDisabledUntil && held.potion !== 'harming') {
            pushEvent(`🛡️ ${peer.username} blocked ${state.username}'s ${held.itemName}!`)
          } else {
            applyDamage(peer, rDmg, { bypassArmor: held.potion === 'harming' })
            try {
              peer.client.write('entity_velocity', {
                entityId: peer.id,
                velocity: {
                  x: Math.round((dx / dist) * 2800),
                  y: 1200,
                  z: Math.round((dz / dist) * 2800),
                },
              })
            } catch {}
            pushEvent(`🏹 ${state.username} hit ${peer.username} with ${held.itemName} (HP=${peer.hp.toFixed(1)})`)
          }
        }
      }
    })

    client.on('block_dig', (d) => {
      // status 5 = release_use_item (lower shield)
      if (d.status === 5) state.shieldUp = false
    })

    // Vanilla keep-alive: without it Mineflayer drops the connection with KeepAliveTimeout
    let keepAliveTimer = null
    const startKeepAlive = () => {
      if (keepAliveTimer) return
      keepAliveTimer = setInterval(() => {
        try {
          client.write('keep_alive', { keepAliveId: BigInt(Date.now()) })
        } catch {}
      }, 10000)
    }
    startKeepAlive()
    client.on('keep_alive', () => {
      state.lastKeepAlive = Date.now()
    })
    client.on('chat', (d) => {
      const msg = d.message || ''
      if (msg.startsWith('/')) handleCommand(state, msg)
    })

    client.on('use_entity', (d) => {
      if (d.mouse !== 1) return // 1 = left-click attack
      const target = players.get(d.target)
      if (!target) return
      const now = Date.now()
      state.swingUntil = now + 260

      const dx = target.x - state.x
      const dz = target.z - state.z
      const dist = Math.hypot(dx, dz)
      if (dist > 4.0) return

      // 1.16.5 Weapon Attack Cooldown scaling (0.2 .. 1.0)
      const elapsed = now - state.lastAttack
      const charge = Math.min(1.0, Math.max(0.25, elapsed / Math.max(200, state.weapon.cdMs)))
      state.lastAttack = now

      // Shield interaction!
      if (target.shieldUp && now >= target.shieldDisabledUntil) {
        if (state.weapon.axe) {
          target.shieldUp = false
          target.shieldDisabledUntil = now + 5000
          pushEvent(`🪓 ${state.username} BROKE ${target.username}'s SHIELD with ${state.weapon.itemName}!`)
        } else {
          pushEvent(`🛡️ ${target.username} blocked ${state.username}'s ${state.weapon.itemName}`)
        }
        return
      }

      if (now - target.lastHurt < 450) return
      target.lastHurt = now

      const isCrit = !state.onGround && state.y > 65.12 && !state.sprinting && charge > 0.85
      const strBonus = now < state.strengthUntil ? 3.0 : 0.0
      const rawDmg = (state.weapon.dmg + strBonus) * (0.2 + 0.8 * charge * charge) * (isCrit ? 1.5 : 1.0)
      const dealt = applyDamage(target, rawDmg)

      if (state.weapon.fire > 0) {
        target.fireUntil = Math.max(target.fireUntil, now + state.weapon.fire)
      }

      const len = dist || 1
      const kbScale = state.weapon.kb * (state.sprinting ? 1.35 : 1.0)
      try {
        target.client.write('entity_velocity', {
          entityId: target.id,
          velocity: {
            x: Math.round((dx / len) * 2900 * kbScale),
            y: 1550,
            z: Math.round((dz / len) * 2900 * kbScale),
          },
        })
      } catch {}

      pushEvent(
        `⚔️ ${state.username} [${state.weapon.itemName}${isCrit ? ' CRIT' : ''}] -> ${target.username} (-${dealt.toFixed(1)} HP | ${target.hp.toFixed(1)}/20)`
      )
    })

    client.on('end', () => {
      if (keepAliveTimer) { clearInterval(keepAliveTimer); keepAliveTimer = null }
      players.delete(entityId)
      broadcastExcept(entityId, 'entity_destroy', { entityIds: [entityId] })
    })
  })

  // Periodic environmental & status tick (Regeneration, Fire, Lava blocks)
  const worldTick = setInterval(() => {
    const now = Date.now()
    for (const p of players.values()) {
      if (now < p.regenUntil && p.hp > 0 && p.hp < 20) {
        p.hp = Math.min(20, p.hp + 0.7)
        syncHealth(p)
      }
      // Check if standing in placed lava or water block
      const bx = Math.round(p.x)
      const bz = Math.round(p.z)
      const footBlk = arenaBlocks.get(`${bx},65,${bz}`)
      if (footBlk && footBlk.name === 'lava') {
        p.fireUntil = Math.max(p.fireUntil, now + 3000)
        applyDamage(p, 1.2)
      } else if (footBlk && footBlk.name === 'water') {
        p.fireUntil = 0
      }
      if (now < p.fireUntil) {
        applyDamage(p, 0.35, { bypassArmor: true })
      }
    }
  }, 250)

  // Live 3D 8-Camera 360° Browser Spectator Dashboard (bound to 0.0.0.0:webPort)
  const httpServer = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*')
    if (req.url === '/api/state') {
      res.writeHead(200, { 'Content-Type': 'application/json' })
      res.end(JSON.stringify(buildSpectatorState()))
      return
    }
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
    res.end(SPECTATOR_HTML)
  })

  const specWss = new WebSocketServer({ server: httpServer })

  function buildSpectatorState() {
    const now = Date.now()
    const botList = Array.from(players.values()).map((p) => ({
      username: p.username,
      x: p.x,
      y: p.y,
      z: p.z,
      yaw: p.yaw,
      pitch: p.pitch,
      hp: p.hp,
      absorption: p.absorption,
      weapon: p.weapon.itemName,
      armorPoints: p.armorPoints,
      shieldUp: Boolean(p.shieldUp && now >= p.shieldDisabledUntil),
      shieldBroken: now < p.shieldDisabledUntil,
      onFire: now < p.fireUntil,
      buffed: now < p.strengthUntil || now < p.speedUntil,
      swinging: now < p.swingUntil,
      hasTotem: p.hasTotem,
    }))
    return {
      ts: now,
      bots: botList,
      blocks: Array.from(arenaBlocks.values()),
      events: combatEvents.slice(-8),
    }
  }

  const broadcastTimer = setInterval(() => {
    if (specWss.clients.size === 0) return
    const payload = JSON.stringify(buildSpectatorState())
    for (const ws of specWss.clients) {
      if (ws.readyState === 1) {
        try { ws.send(payload) } catch {}
      }
    }
  }, 80)

  httpServer.listen(webPort, '0.0.0.0', () => {
    console.log(`Live 3D 360° Spectator Arena Dashboard listening on http://0.0.0.0:${webPort}`)
  })

  server.on('close', () => {
    clearInterval(worldTick)
    clearInterval(broadcastTimer)
    try { httpServer.close() } catch {}
  })

  return { server, httpServer }
}

const SPECTATOR_HTML = `<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<title>Live Minecraft 1.16.5 PvP Server — 8-Cam 360° Arena</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0b0e16; color: #e8f0ff; font-family: system-ui, -apple-system, sans-serif; overflow-x: hidden; }
  header { display: flex; justify-content: space-between; align-items: center; padding: 10px 18px; background: #121829; border-bottom: 2px solid #2b3c63; }
  header h1 { font-size: 18px; color: #f0f6ff; }
  header .badge { background: #16a34a; color: #fff; font-weight: 700; font-size: 12px; padding: 4px 10px; border-radius: 999px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 8px 12px; }
  .cam-box { position: relative; background: #151b2b; border: 2px solid #38bdf8; border-radius: 6px; overflow: hidden; aspect-ratio: 16 / 11; }
  .cam-box.enemy { border-color: #f87171; }
  .cam-box canvas { width: 100%; height: 100%; display: block; }
  .cam-label { position: absolute; top: 4px; left: 6px; right: 6px; display: flex; justify-content: space-between; font-size: 11px; font-weight: 700; background: rgba(11,14,22,0.82); padding: 3px 7px; border-radius: 4px; color: #7dd3fc; pointer-events: none; }
  .cam-box.enemy .cam-label { color: #fca5a5; }
  .hud { display: grid; grid-template-columns: 1fr 1fr 1.2fr; gap: 12px; padding: 6px 12px 12px; }
  .card { background: #121829; border: 1px solid #2a395c; border-radius: 6px; padding: 10px 14px; font-size: 13px; }
  .card h3 { font-size: 14px; margin-bottom: 6px; }
  .bar { height: 12px; background: #252e42; border-radius: 4px; overflow: hidden; margin-top: 6px; }
  .bar > div { height: 100%; transition: width 0.12s linear; }
  .events { font-family: ui-monospace, monospace; font-size: 12px; color: #cbd5e1; line-height: 1.45; max-height: 115px; overflow: hidden; }
</style>
</head>
<body>
<header>
  <div>
    <h1>⚔️ LIVE MINECRAFT 1.16.5 PvP SERVER — 360° MULTI-CAM SPECTATOR</h1>
    <div style="font-size:12px;color:#94a3b8;">Ряд 1: 4 камеры 360° на LearnerBot (PPO 50k) &nbsp;|&nbsp; Ряд 2: 4 камеры 360° на EnemyBot (Асимметричный рандом + блоки)</div>
  </div>
  <div id="statusBadge" class="badge">LIVE SERVER :25565</div>
</header>

<div class="grid" id="camGrid"></div>

<div class="hud">
  <div class="card">
    <h3 style="color:#38bdf8;">🟦 LearnerBot (PPO 50k Neural Policy)</h3>
    <div id="lMeta">Ожидание подключения бота...</div>
    <div class="bar"><div id="lBar" style="width:100%;background:#38bdf8;"></div></div>
  </div>
  <div class="card">
    <h3 style="color:#f87171;">🟥 EnemyBot (Asymmetric Chaos Bot)</h3>
    <div id="eMeta">Ожидание подключения бота...</div>
    <div class="bar"><div id="eBar" style="width:100%;background:#f87171;"></div></div>
  </div>
  <div class="card">
    <h3 style="color:#fbbf24;">📜 Живой лог реального сервера (удары, блоки, ловушки)</h3>
    <div class="events" id="evLog">Сервер запущен. Ожидание боя...</div>
  </div>
</div>

<script>
const CAMS = [
  { bot: 'LearnerBot', base: 0, title: 'L-CAM 1 [FRONT + 360°]', cls: '' },
  { bot: 'LearnerBot', base: 90, title: 'L-CAM 2 [RIGHT + 360°]', cls: '' },
  { bot: 'LearnerBot', base: 180, title: 'L-CAM 3 [REVERSE + 360°]', cls: '' },
  { bot: 'LearnerBot', base: 270, title: 'L-CAM 4 [LEFT + 360°]', cls: '' },
  { bot: 'EnemyBot', base: 0, title: 'E-CAM 1 [FRONT + 360°]', cls: 'enemy' },
  { bot: 'EnemyBot', base: 90, title: 'E-CAM 2 [RIGHT + 360°]', cls: 'enemy' },
  { bot: 'EnemyBot', base: 180, title: 'E-CAM 3 [REVERSE + 360°]', cls: 'enemy' },
  { bot: 'EnemyBot', base: 270, title: 'E-CAM 4 [LEFT + 360°]', cls: 'enemy' },
];
const grid = document.getElementById('camGrid');
const ctxs = CAMS.map((c, idx) => {
  const box = document.createElement('div');
  box.className = 'cam-box ' + c.cls;
  box.innerHTML = '<div class="cam-label"><span>' + c.title + '</span><span id="az'+idx+'">000°</span></div><canvas width="420" height="288"></canvas>';
  grid.appendChild(box);
  return { ...c, idx, canvas: box.querySelector('canvas'), ctx: box.querySelector('canvas').getContext('2d'), azEl: box.querySelector('#az'+idx) };
});

let state = { bots: [], blocks: [], events: [] };
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(proto + '//' + location.host);
  ws.onmessage = (ev) => { try { state = JSON.parse(ev.data); updateHUD(); } catch {} };
  ws.onclose = () => setTimeout(connectWS, 1500);
}
connectWS();

function updateHUD() {
  const L = state.bots.find(b => b.username === 'LearnerBot');
  const E = state.bots.find(b => b.username === 'EnemyBot');
  if (L) {
    document.getElementById('lMeta').textContent = 'Оружие: ' + L.weapon + ' | Броня: ' + L.armorPoints + ' ед. | HP: ' + L.hp.toFixed(1) + '/20 (+' + L.absorption.toFixed(1) + ')' + (L.shieldUp ? ' [ЩИТ]' : '') + (L.onFire ? ' [ГОРИТ]' : '');
    document.getElementById('lBar').style.width = Math.max(0, Math.min(100, L.hp * 5)) + '%';
  }
  if (E) {
    document.getElementById('eMeta').textContent = 'Оружие: ' + E.weapon + ' | Броня: ' + E.armorPoints + ' ед. | HP: ' + E.hp.toFixed(1) + '/20 (+' + E.absorption.toFixed(1) + ')' + (E.shieldUp ? ' [ЩИТ]' : '') + (E.shieldBroken ? ' [ЩИТ ПРОБИТ]' : '');
    document.getElementById('eBar').style.width = Math.max(0, Math.min(100, E.hp * 5)) + '%';
  }
  if (state.events && state.events.length) {
    document.getElementById('evLog').innerHTML = state.events.slice().reverse().map(e => '<div>' + e + '</div>').join('');
  }
}

const CUBE_V = [[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]];
const FACES = [[[4,5,6,7],0.95],[[1,0,3,2],0.72],[[5,1,2,6],0.84],[[0,4,7,3],0.78],[[7,6,2,3],1.06]];

function proj(pt, camPos, fwd, right, up, W, H, f) {
  const rx = pt[0]-camPos[0], ry = pt[1]-camPos[1], rz = pt[2]-camPos[2];
  const z = rx*fwd[0] + ry*fwd[1] + rz*fwd[2];
  if (z <= 0.2) return null;
  const x = rx*right[0] + ry*right[1] + rz*right[2];
  const y = rx*up[0] + ry*up[1] + rz*up[2];
  return [W*0.5 + (x/z)*f, H*0.5 - (y/z)*f, z];
}

function addBox(polys, center, size, yaw, rgb, camPos, fwd, right, up, W, H, f) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const wv = CUBE_V.map(v => {
    const sx = v[0]*size[0], sy2 = v[1]*size[1], sz = v[2]*size[2];
    return [center[0] + sx*cy + sz*sy, center[1] + sy2, center[2] - sx*sy + sz*cy];
  });
  const pv = wv.map(p => proj(p, camPos, fwd, right, up, W, H, f));
  if (pv.some(p => !p)) return;
  for (const [q, sh] of FACES) {
    const zAvg = (pv[q[0]][2]+pv[q[1]][2]+pv[q[2]][2]+pv[q[3]][2])*0.25;
    const col = 'rgb(' + Math.min(255,(rgb[0]*sh)|0) + ',' + Math.min(255,(rgb[1]*sh)|0) + ',' + Math.min(255,(rgb[2]*sh)|0) + ')';
    polys.push({ z: zAvg, pts: q.map(i => pv[i]), col });
  }
}

function renderCam(c, orbitDeg) {
  const ctx = c.ctx, W = c.canvas.width, H = c.canvas.height;
  ctx.fillStyle = c.cls ? '#201622' : '#141c2b';
  ctx.fillRect(0, 0, W, H);

  const anchor = state.bots.find(b => b.username === c.bot) || { x: 0, y: 65, z: 0, yaw: 0 };
  const other = state.bots.find(b => b.username !== c.bot) || { x: 2, y: 65, z: 2 };
  const by0 = anchor.y - 65;

  const az = (c.base + orbitDeg) % 360;
  c.azEl.textContent = '360° Az: ' + String(az|0).padStart(3, '0') + '°';
  const th = ((anchor.yaw || 0) * Math.PI / 180) + az * Math.PI / 180;
  const cx = Math.max(-10.5, Math.min(10.5, anchor.x - Math.sin(th) * 4.8));
  const cz = Math.max(-10.5, Math.min(10.5, anchor.z - Math.cos(th) * 4.8));
  const camPos = [cx, by0 + 2.6, cz];
  const look = [anchor.x*0.72 + other.x*0.28, (anchor.y-65)*0.75 + 1.15, anchor.z*0.72 + other.z*0.28];

  let fx = look[0]-camPos[0], fy = look[1]-camPos[1], fz = look[2]-camPos[2];
  const fl = Math.hypot(fx,fy,fz)||1; fx/=fl; fy/=fl; fz/=fl;
  let rx = fz, ry = 0, rz = -fx; // cross(fwd, [0,1,0]) -> [-fz, 0, fx] wait: fwd x up = [fy*0 - fz*1, fz*0 - fx*0, fx*1 - fy*0] = [-fz, 0, fx]
  rx = -fz; rz = fx;
  const rl = Math.hypot(rx,rz)||1; rx/=rl; rz/=rl;
  const ux = ry*fz - rz*fy, uy = rz*fx - rx*fz, uz = rx*fy - ry*fx;
  const fwd=[fx,fy,fz], right=[rx,ry,rz], up=[ux,uy,uz], f = W * 0.78;

  // Floor grid
  for (let gx = -8; gx <= 8; gx += 4) {
    for (let gz = -8; gz <= 8; gz += 4) {
      const q = [[gx-2,0,gz-2],[gx+2,0,gz-2],[gx+2,0,gz+2],[gx-2,0,gz+2]].map(p => proj(p,camPos,fwd,right,up,W,H,f));
      if (q.some(p => !p)) continue;
      ctx.fillStyle = ((gx+gz)/4)%2===0 ? '#3f4f46' : '#35443c';
      ctx.beginPath(); ctx.moveTo(q[0][0],q[0][1]); q.slice(1).forEach(p=>ctx.lineTo(p[0],p[1])); ctx.closePath(); ctx.fill(); ctx.strokeStyle='#27332d'; ctx.stroke();
    }
  }

  const polys = [];
  for (const blk of (state.blocks || [])) {
    const col = blk.name.includes('obsidian') ? [56,38,84] : (blk.name.includes('cobweb') ? [235,240,248] : (blk.name.includes('lava') ? [255,105,25] : (blk.name.includes('water') ? [55,148,255] : [145,148,156])));
    const h = (blk.name.includes('lava') || blk.name.includes('water')) ? 0.25 : 1.0;
    addBox(polys, [blk.x, (blk.y - 65) + h*0.5, blk.z], [1, h, 1], 0, col, camPos, fwd, right, up, W, H, f);
  }

  for (const b of (state.bots || [])) {
    const isL = b.username === 'LearnerBot';
    const baseCol = isL ? [56, 178, 255] : [245, 86, 86];
    const bodyCol = b.armorPoints >= 14 ? [72, 228, 224] : (b.armorPoints >= 8 ? [210, 218, 228] : baseCol);
    const otherB = state.bots.find(o => o.username !== b.username) || { x: 0, z: 0 };
    const yaw = Math.atan2(otherB.x - b.x, otherB.z - b.z);
    const y0 = b.y - 65;
    addBox(polys, [b.x, y0 + 0.38, b.z], [0.46, 0.75, 0.26], yaw, bodyCol, camPos, fwd, right, up, W, H, f);
    addBox(polys, [b.x, y0 + 1.12, b.z], [0.54, 0.72, 0.30], yaw, bodyCol, camPos, fwd, right, up, W, H, f);
    addBox(polys, [b.x, y0 + 1.74, b.z], [0.48, 0.48, 0.48], yaw, baseCol, camPos, fwd, right, up, W, H, f);
    addBox(polys, [b.x + Math.sin(yaw)*0.55, y0 + 1.1, b.z + Math.cos(yaw)*0.55], [0.14, 0.18, 0.72], yaw, [225,232,240], camPos, fwd, right, up, W, H, f);
    if (b.shieldUp) {
      addBox(polys, [b.x + Math.sin(yaw)*0.45, y0 + 1.15, b.z + Math.cos(yaw)*0.45], [0.65, 0.85, 0.10], yaw, [215, 170, 85], camPos, fwd, right, up, W, H, f);
    }
  }

  polys.sort((a,b) => b.z - a.z);
  for (const p of polys) {
    ctx.fillStyle = p.col;
    ctx.beginPath(); ctx.moveTo(p.pts[0][0], p.pts[0][1]);
    for (let i=1;i<p.pts.length;i++) ctx.lineTo(p.pts[i][0], p.pts[i][1]);
    ctx.closePath(); ctx.fill(); ctx.strokeStyle = '#141824'; ctx.lineWidth = 1; ctx.stroke();
  }
}

function animate(ts) {
  const orbitDeg = ((ts || 0) * 0.032) % 360;
  for (const c of ctxs) renderCam(c, orbitDeg);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
</script>
</body>
</html>`

if (require.main === module) {
  const mcPort = Number(process.env.MC_PORT || 25565)
  const webPort = Number(process.env.WEB_PORT || 8080)
  startPvPServer({ mcPort, webPort })
  console.log(`Live 1.16.5 PvP Arena Server listening on 127.0.0.1:${mcPort}`)
}

module.exports = { startPvPServer, startMockServer: (port, host) => startPvPServer({ mcPort: port, mcHost: host, webPort: 0 }).server }

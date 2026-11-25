// Minecraft PvP AI — Randomizer / Asymmetric Chaos Loadout Mode (1.16.5)
// Every episode, LearnerBot and EnemyBot receive COMPLETELY DIFFERENT, randomly rolled
// inventories: random weapons/tools/items, asymmetric armor pieces, building/trap blocks
// (cobblestone, obsidian, cobwebs, lava/water buckets), shields/totems, and consumables!

const mineflayer = require('mineflayer')
const { WebSocketServer } = require('ws')
const { Vec3 } = require('vec3')

const HOST = process.env.MC_HOST || '127.0.0.1'
const PORT = Number(process.env.MC_PORT || 25565)
const WS_PORT = Number(process.env.WS_PORT || 8765)
const MC_VERSION = process.env.MC_VERSION || '1.16.5'
const STEP_DELAY_MS = Number(process.env.STEP_DELAY_MS || 100)
const MAX_STEPS = Number(process.env.MAX_STEPS || 600)

// Full random melee item pool (from sticks & hoes to enchanted netherite axes)
const MELEE_POOL = [
  { id: 'stick_kb', item: 'stick', cmd: 'minecraft:stick{Enchantments:[{id:"minecraft:knockback",lvl:2}]}', dmg: 1.0, cdMs: 250, kb: 2.2, shieldBreak: false, fire: false },
  { id: 'wooden_shovel', item: 'wooden_shovel', cmd: 'minecraft:wooden_shovel', dmg: 2.5, cdMs: 1000, kb: 1.0, shieldBreak: false, fire: false },
  { id: 'golden_hoe', item: 'golden_hoe', cmd: 'minecraft:golden_hoe{Enchantments:[{id:"minecraft:fire_aspect",lvl:1}]}', dmg: 1.0, cdMs: 250, kb: 1.0, shieldBreak: false, fire: true },
  { id: 'stone_sword', item: 'stone_sword', cmd: 'minecraft:stone_sword', dmg: 5.0, cdMs: 625, kb: 1.0, shieldBreak: false, fire: false },
  { id: 'iron_pickaxe', item: 'iron_pickaxe', cmd: 'minecraft:iron_pickaxe', dmg: 4.0, cdMs: 830, kb: 1.0, shieldBreak: false, fire: false },
  { id: 'iron_sword', item: 'iron_sword', cmd: 'minecraft:iron_sword', dmg: 6.0, cdMs: 625, kb: 1.0, shieldBreak: false, fire: false },
  { id: 'iron_axe', item: 'iron_axe', cmd: 'minecraft:iron_axe', dmg: 9.0, cdMs: 1100, kb: 1.1, shieldBreak: true, fire: false },
  { id: 'diamond_sword', item: 'diamond_sword', cmd: 'minecraft:diamond_sword', dmg: 7.0, cdMs: 625, kb: 1.0, shieldBreak: false, fire: false },
  { id: 'trident', item: 'trident', cmd: 'minecraft:trident{Enchantments:[{id:"minecraft:impaling",lvl:2}]}', dmg: 9.0, cdMs: 910, kb: 1.1, shieldBreak: false, fire: false },
  { id: 'diamond_axe', item: 'diamond_axe', cmd: 'minecraft:diamond_axe{Enchantments:[{id:"minecraft:sharpness",lvl:2}]}', dmg: 10.5, cdMs: 1000, kb: 1.1, shieldBreak: true, fire: false },
  { id: 'netherite_sword_fire', item: 'netherite_sword', cmd: 'minecraft:netherite_sword{Enchantments:[{id:"minecraft:fire_aspect",lvl:2}]}', dmg: 8.0, cdMs: 625, kb: 1.2, shieldBreak: false, fire: true },
  { id: 'netherite_axe_sharp', item: 'netherite_axe', cmd: 'minecraft:netherite_axe{Enchantments:[{id:"minecraft:sharpness",lvl:3}]}', dmg: 12.0, cdMs: 1000, kb: 1.2, shieldBreak: true, fire: false },
]

const ARMOR_TIERS = [
  { name: 'none', cmds: [], reduction: 0.0 },
  { name: 'leather', cmds: ['leather_chestplate', 'leather_boots'], reduction: 0.16 },
  { name: 'chain_gold', cmds: ['golden_helmet', 'chainmail_chestplate', 'chainmail_leggings'], reduction: 0.36 },
  { name: 'iron', cmds: ['iron_helmet', 'iron_chestplate', 'iron_leggings', 'iron_boots'], reduction: 0.56 },
  { name: 'diamond_mix', cmds: ['diamond_helmet', 'diamond_chestplate', 'iron_leggings', 'diamond_boots'], reduction: 0.72 },
]

const BLOCK_TYPES = ['minecraft:cobblestone', 'minecraft:oak_planks', 'minecraft:obsidian', 'minecraft:hay_block']
const RANGED_TYPES = [
  { id: 'snowball', item: 'snowball', cmd: 'minecraft:snowball', count: [4, 12] },
  { id: 'egg', item: 'egg', cmd: 'minecraft:egg', count: [4, 12] },
  { id: 'crossbow', item: 'crossbow', cmd: 'minecraft:crossbow{Charged:1b,ChargedProjectiles:[{id:"minecraft:arrow",Count:1b}]}', count: [1, 3] },
  { id: 'splash_harming', item: 'splash_potion', cmd: 'minecraft:splash_potion{Potion:"minecraft:harming"}', count: [1, 2] },
]

const HEAL_TYPES = [
  { id: 'cooked_beef', cmd: 'minecraft:cooked_beef', healEq: 3.0, count: [2, 5] },
  { id: 'golden_carrot', cmd: 'minecraft:golden_carrot', healEq: 4.0, count: [2, 5] },
  { id: 'suspicious_stew', cmd: 'minecraft:suspicious_stew', healEq: 5.0, count: [1, 2] },
  { id: 'golden_apple', cmd: 'minecraft:golden_apple', healEq: 7.5, count: [1, 3] },
  { id: 'splash_heal', cmd: 'minecraft:splash_potion{Potion:"minecraft:strong_healing"}', healEq: 8.0, count: [1, 3] },
]

function randInt(lo, hi) {
  return lo + Math.floor(Math.random() * (hi - lo + 1))
}

function pickDifferentIndex(len, avoidIdx) {
  if (len <= 1) return 0
  let idx = Math.floor(Math.random() * len)
  while (idx === avoidIdx) {
    idx = Math.floor(Math.random() * len)
  }
  return idx
}

function rollRandomLoadout(avoidWeaponIdx = -1, avoidArmorIdx = -1) {
  const weaponIdx = avoidWeaponIdx >= 0
    ? pickDifferentIndex(MELEE_POOL.length, avoidWeaponIdx)
    : Math.floor(Math.random() * MELEE_POOL.length)
  const armorIdx = avoidArmorIdx >= 0
    ? pickDifferentIndex(ARMOR_TIERS.length, avoidArmorIdx)
    : Math.floor(Math.random() * ARMOR_TIERS.length)

  const weapon = MELEE_POOL[weaponIdx]
  const armor = ARMOR_TIERS[armorIdx]
  const blockItem = BLOCK_TYPES[Math.floor(Math.random() * BLOCK_TYPES.length)]
  const blocks = Math.random() < 0.85 ? randInt(4, 24) : 0
  const cobwebs = Math.random() < 0.55 ? randInt(1, 4) : 0
  const lavaBuckets = Math.random() < 0.45 ? 1 : 0
  const waterBuckets = Math.random() < 0.50 ? 1 : 0
  const hasShield = Math.random() < 0.55
  const hasTotem = !hasShield && Math.random() < 0.20

  const healSpec = HEAL_TYPES[Math.floor(Math.random() * HEAL_TYPES.length)]
  const heals = Math.random() < 0.75 ? randInt(healSpec.count[0], healSpec.count[1]) : 0
  const gaps = healSpec.id === 'golden_apple' ? heals : (Math.random() < 0.45 ? randInt(1, 2) : 0)
  const splashes = healSpec.id !== 'golden_apple' ? heals : (Math.random() < 0.45 ? randInt(1, 2) : 0)

  const pearls = Math.random() < 0.55 ? randInt(1, 3) : 0
  const buffs = Math.random() < 0.50 ? randInt(1, 2) : 0
  const rangedSpec = RANGED_TYPES[Math.floor(Math.random() * RANGED_TYPES.length)]
  const ranged = Math.random() < 0.75 ? randInt(rangedSpec.count[0], rangedSpec.count[1]) : 0

  // PRO explosive meta (level with the trainer/opponent loadouts)
  const crystals = Math.random() < 0.62 ? randInt(1, 6) : 0
  const obsidian = Math.random() < 0.78 ? randInt(4, 16) : 0
  const anchors = Math.random() < 0.42 ? randInt(1, 3) : 0
  const minecarts = Math.random() < 0.46 ? randInt(1, 4) : 0

  return {
    weaponIdx,
    armorIdx,
    summary: `${weapon.id}+${armor.name}+blk(${blocks})`,
    weapon,
    armor,
    blockItem,
    blocks,
    cobwebs,
    lavaBuckets,
    waterBuckets,
    hasShield,
    hasTotem,
    gaps,
    splashes,
    pearls,
    buffs,
    rangedSpec,
    ranged,
    crystals,
    obsidian,
    anchors,
    minecarts,
    cageUntil: 0,
    shieldUp: false,
    shieldDisabledUntil: 0,
    buffUntil: 0,
    regenUntil: 0,
    pearlCdUntil: 0,
    hazardUntil: 0,
  }
}

let learner = null
let enemy = null
let learnerSpawned = false
let enemySpawned = false
let learnerReconnecting = false
let enemyReconnecting = false

let lLoad = rollRandomLoadout()
let eLoad = rollRandomLoadout(lLoad.weaponIdx, lLoad.armorIdx)

let lastAttackTime = 0
let lastEnemyAttackTime = 0
let prevLearnerHP = 20
let prevEnemyHP = 20
let obstacleCount = 0

let episodeDone = false
let resetting = false
let stepCount = 0
let arenaDone = false
let episodeLiveAt = 0

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v)
}

function installMovementGuard(bot, label) {
  if (!bot || !bot._client || bot._movementGuardInstalled) return
  bot._movementGuardInstalled = true
  const origWrite = bot._client.write.bind(bot._client)
  let warned = false
  bot._client.write = (name, params) => {
    if (
      params &&
      (name === 'position' || name === 'position_look' || name === 'look')
    ) {
      for (const field of ['x', 'y', 'z', 'yaw', 'pitch']) {
        if (field in params && !isFiniteNumber(params[field])) {
          if (!warned) {
            warned = true
            console.warn(
              `[${label}] Blocked invalid ${name} packet (${field}=${params[field]})`
            )
          }
          if (bot.entity) {
            if (!isFiniteNumber(bot.entity.velocity.x)) bot.entity.velocity.x = 0
            if (!isFiniteNumber(bot.entity.velocity.y)) bot.entity.velocity.y = 0
            if (!isFiniteNumber(bot.entity.velocity.z)) bot.entity.velocity.z = 0
            if (!isFiniteNumber(bot.entity.yaw)) bot.entity.yaw = 0
            if (!isFiniteNumber(bot.entity.pitch)) bot.entity.pitch = 0
          }
          return
        }
      }
    }
    return origWrite(name, params)
  }
}

function getPeerEntity(viewerBot, targetBot) {
  if (!viewerBot || !targetBot || !targetBot.username) return null
  const playerEntry = viewerBot.players && viewerBot.players[targetBot.username]
  if (playerEntry && playerEntry.entity && playerEntry.entity.isValid !== false) {
    return playerEntry.entity
  }
  if (viewerBot.entities) {
    for (const ent of Object.values(viewerBot.entities)) {
      if (ent && ent.username === targetBot.username && ent.isValid !== false) {
        return ent
      }
    }
  }
  return null
}

function botsReady() {
  return Boolean(
    learner &&
      enemy &&
      learnerSpawned &&
      enemySpawned &&
      learner.entity &&
      enemy.entity
  )
}

function safeLookAt(bot, point) {
  if (!bot || !bot.entity || !point) return
  const dx = point.x - bot.entity.position.x
  const dz = point.z - bot.entity.position.z
  if (!isFiniteNumber(dx) || !isFiniteNumber(dz) || dx * dx + dz * dz < 1e-6) {
    return
  }
  bot.lookAt(point, true).catch(() => {})
}

function equipItemByKeyword(bot, keyword, dest = 'hand') {
  if (!bot || !bot.inventory) return false
  const item = bot.inventory
    .items()
    .find((it) => it && it.name && it.name.includes(keyword))
  if (item) {
    bot.equip(item, dest).catch(() => {})
    return true
  }
  return false
}

function equipMainWeapon(bot, load) {
  if (!bot || !load) return
  equipItemByKeyword(bot, load.weapon.item, 'hand')
}

function equipArmorSet(bot) {
  if (!bot || !bot.inventory) return
  for (const item of bot.inventory.items()) {
    if (!item || !item.name) continue
    if (item.name.endsWith('_helmet')) bot.equip(item, 'head').catch(() => {})
    else if (item.name.endsWith('_chestplate')) bot.equip(item, 'torso').catch(() => {})
    else if (item.name.endsWith('_leggings')) bot.equip(item, 'legs').catch(() => {})
    else if (item.name.endsWith('_boots')) bot.equip(item, 'feet').catch(() => {})
  }
}

function clearControls(bot) {
  if (!bot) return
  for (const c of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
    try { bot.setControlState(c, false) } catch {}
  }
}

function buildGiveCommands(username, load) {
  const cmds = [`/give ${username} ${load.weapon.cmd} 1`]
  for (const piece of load.armor.cmds) {
    cmds.push(`/give ${username} minecraft:${piece} 1`)
  }
  if (load.hasShield) cmds.push(`/give ${username} minecraft:shield 1`)
  if (load.hasTotem) cmds.push(`/give ${username} minecraft:totem_of_undying 1`)
  if (load.blocks > 0) cmds.push(`/give ${username} ${load.blockItem} ${load.blocks}`)
  if (load.cobwebs > 0) cmds.push(`/give ${username} minecraft:cobweb ${load.cobwebs}`)
  if (load.lavaBuckets > 0) cmds.push(`/give ${username} minecraft:lava_bucket 1`)
  if (load.waterBuckets > 0) cmds.push(`/give ${username} minecraft:water_bucket 1`)
  if (load.gaps > 0) cmds.push(`/give ${username} minecraft:golden_apple ${load.gaps}`)
  if (load.splashes > 0) {
    cmds.push(`/give ${username} minecraft:splash_potion{Potion:"minecraft:strong_healing"} ${load.splashes}`)
  }
  if (load.pearls > 0) cmds.push(`/give ${username} minecraft:ender_pearl ${load.pearls}`)
  if (load.ranged > 0) cmds.push(`/give ${username} ${load.rangedSpec.cmd} ${load.ranged}`)
  // PRO meta gear: end crystals, respawn anchors, TNT minecarts, obsidian
  if (load.crystals > 0) cmds.push(`/give ${username} minecraft:end_crystal ${load.crystals}`)
  if (load.anchors > 0) cmds.push(`/give ${username} minecraft:respawn_anchor ${load.anchors}`)
  if (load.minecarts > 0) cmds.push(`/give ${username} minecraft:tnt_minecart ${load.minecarts}`)
  if (load.obsidian > 0) cmds.push(`/give ${username} minecraft:obsidian ${load.obsidian}`)
  return cmds
}

function resetEpisode() {
  if (!botsReady()) return false
  if (resetting) return true
  resetting = true
  clearControls(learner)
  clearControls(enemy)
  try { learner.deactivateItem() } catch {}
  try { enemy.deactivateItem() } catch {}
  episodeDone = true
  obstacleCount = 0

  // Roll completely independent, asymmetric random inventories for each bot!
  lLoad = rollRandomLoadout()
  eLoad = rollRandomLoadout(lLoad.weaponIdx, lLoad.armorIdx)

  const cmds = [
    // Clean any placed blocks, cobwebs, water, or lava inside the 20x20 arena!
    '/fill -9 65 -9 9 72 9 minecraft:air',
    '/clear LearnerBot',
    '/clear EnemyBot',
    '/effect clear LearnerBot',
    '/effect clear EnemyBot',
    ...buildGiveCommands('LearnerBot', lLoad),
    ...buildGiveCommands('EnemyBot', eLoad),
    '/effect give LearnerBot minecraft:instant_health 1 10 true',
    '/effect give EnemyBot minecraft:instant_health 1 10 true',
    '/effect give LearnerBot minecraft:saturation 1 10 true',
    '/effect give EnemyBot minecraft:saturation 1 10 true',
  ]

  cmds.forEach((cmd, i) => setTimeout(() => {
    try { if (learner && learnerSpawned) learner.chat(cmd) } catch {}
  }, i * 65))

  const readyDelay = cmds.length * 65 + 250
  setTimeout(() => {
    equipArmorSet(learner)
    equipArmorSet(enemy)
    if (lLoad.hasShield) equipItemByKeyword(learner, 'shield', 'off-hand')
    else if (lLoad.hasTotem) equipItemByKeyword(learner, 'totem', 'off-hand')
    if (eLoad.hasShield) equipItemByKeyword(enemy, 'shield', 'off-hand')
    else if (eLoad.hasTotem) equipItemByKeyword(enemy, 'totem', 'off-hand')
    equipMainWeapon(learner, lLoad)
    equipMainWeapon(enemy, eLoad)
  }, readyDelay)

  setTimeout(() => {
    prevLearnerHP = learner && learner.health !== undefined ? learner.health : 20
    prevEnemyHP = enemy && enemy.health !== undefined ? enemy.health : 20
    episodeDone = false
    resetting = false
    stepCount = 0
    episodeLiveAt = Date.now()
  }, readyDelay + 200)
  return true
}

function clamp(v, lo, hi) {
  if (!isFiniteNumber(v)) return 0
  return Math.max(lo, Math.min(hi, v))
}

function placeBlockBetween(actorBot, targetPos, blockId) {
  if (!actorBot || !actorBot.entity || !learner || !learnerSpawned) return false
  const ap = actorBot.entity.position
  const mx = Math.round((ap.x + targetPos.x) * 0.5)
  const mz = Math.round((ap.z + targetPos.z) * 0.5)
  const bx = Math.max(-8, Math.min(8, mx))
  const bz = Math.max(-8, Math.min(8, mz))
  try {
    learner.chat(`/setblock ${bx} 65 ${bz} ${blockId} replace`)
    if (!blockId.includes('cobweb') && !blockId.includes('lava') && !blockId.includes('water')) {
      learner.chat(`/setblock ${bx} 66 ${bz} ${blockId} replace`)
    }
    return true
  } catch {
    return false
  }
}

function getObservation() {
  if (!botsReady()) {
    return new Array(30).fill(0)
  }
  const now = Date.now()
  const lp = learner.entity.position
  const ep = enemy.entity.position
  const dx = ep.x - lp.x
  const dz = ep.z - lp.z
  const dist = Math.sqrt(dx * dx + dz * dz)
  const cooldownReady = (now - lastAttackTime >= lLoad.weapon.cdMs) ? 1 : 0
  const lVel = learner.entity.velocity ? learner.entity.velocity.norm() : 0
  const eVel = enemy.entity.velocity ? enemy.entity.velocity.norm() : 0
  return [
    clamp((learner.health ?? 20) / 20, 0, 1.5),
    clamp((enemy.health ?? 20) / 20, 0, 1.5),
    clamp(dist / 20, 0, 1),
    clamp(lVel, 0, 2),
    clamp(eVel, 0, 2),
    clamp(dx / 20, -1, 1),
    clamp(dz / 20, -1, 1),
    learner.entity.onGround ? 1 : 0,
    cooldownReady,
    clamp(stepCount / MAX_STEPS, 0, 1),
    clamp(lLoad.weapon.dmg / 14, 0, 1),
    clamp(lLoad.weapon.cdMs / 1200, 0, 1),
    clamp(eLoad.weapon.dmg / 14, 0, 1),
    clamp(lLoad.armor.reduction, 0, 1),
    clamp(eLoad.armor.reduction, 0, 1),
    clamp(lLoad.gaps / 5, 0, 1),
    clamp(lLoad.splashes / 5, 0, 1),
    clamp(lLoad.pearls / 4, 0, 1),
    clamp((lLoad.buffs + lLoad.ranged) / 12, 0, 1),
    (lLoad.hasShield && lLoad.shieldUp && now >= lLoad.shieldDisabledUntil) ? 1 : 0,
    (eLoad.hasShield && eLoad.shieldUp && now >= eLoad.shieldDisabledUntil) ? 1 : 0,
    clamp(lLoad.blocks / 24, 0, 1),
    clamp((lLoad.cobwebs + lLoad.lavaBuckets + lLoad.waterBuckets) / 6, 0, 1),
    clamp(obstacleCount / 6, 0, 1),
    clamp((lLoad.crystals || 0) / 6, 0, 1),
    clamp((lLoad.anchors || 0) / 4, 0, 1),
    clamp((lLoad.minecarts || 0) / 4, 0, 1),
    clamp(((eLoad.crystals || 0) + (eLoad.anchors || 0) + (eLoad.minecarts || 0)) / 10, 0, 1),
    clamp((lLoad.obsidian || 0) / 16, 0, 1),
    now < (lLoad.cageUntil || 0) ? 1 : 0,
  ]
}

function doAction(action) {
  if (!botsReady()) {
    return { usedConsumable: null, wastedConsumable: false, brokeEnemyShield: false, placedBlock: null }
  }
  const now = Date.now()
  clearControls(learner)
  if (lLoad.shieldUp && action !== 12) {
    lLoad.shieldUp = false
    try { learner.deactivateItem() } catch {}
  }

  const targetEntity = getPeerEntity(learner, enemy)
  const targetPos = (targetEntity ? targetEntity.position : enemy.entity.position).offset(0, 1.5, 0)
  safeLookAt(learner, targetPos)

  let usedConsumable = null
  let wastedConsumable = false
  let brokeEnemyShield = false
  let placedBlock = null

  const tryAttack = () => {
    if (now - lastAttackTime >= lLoad.weapon.cdMs - 25 && targetEntity) {
      const dist = learner.entity.position.distanceTo(targetEntity.position)
      if (dist <= 3.5) {
        try {
          equipMainWeapon(learner, lLoad)
          learner.attack(targetEntity)
          lastAttackTime = now
          if (lLoad.weapon.shieldBreak && eLoad.shieldUp && now >= eLoad.shieldDisabledUntil) {
            eLoad.shieldUp = false
            eLoad.shieldDisabledUntil = now + 5000
            brokeEnemyShield = true
            try { enemy.deactivateItem() } catch {}
          }
        } catch {}
      }
    }
  }

  switch (action) {
    case 1: learner.setControlState('forward', true); break
    case 2: learner.setControlState('back', true); break
    case 3: learner.setControlState('left', true); break
    case 4: learner.setControlState('right', true); break
    case 5: learner.setControlState('forward', true); tryAttack(); break
    case 6: learner.setControlState('left', true); tryAttack(); break
    case 7: learner.setControlState('right', true); tryAttack(); break
    case 8: learner.setControlState('jump', true); break
    case 9:
      learner.setControlState('forward', true)
      learner.setControlState('jump', true)
      learner.setControlState('sprint', true)
      tryAttack()
      break
    case 10:
      learner.setControlState('forward', true)
      learner.setControlState('sprint', true)
      break
    case 11: learner.setControlState('back', true); tryAttack(); break
    case 12:
      if (lLoad.hasShield && now >= lLoad.shieldDisabledUntil) {
        lLoad.shieldUp = true
        learner.setControlState('back', true)
        try { learner.activateItem(true) } catch {}
      } else {
        wastedConsumable = true
      }
      break
    case 13:
      if (lLoad.gaps > 0) {
        const hp = learner.health ?? 20
        if (hp >= 19.5 && now < lLoad.regenUntil) wastedConsumable = true
        lLoad.gaps--
        lLoad.regenUntil = now + 5000
        usedConsumable = 'golden_apple'
        try {
          learner.chat('/effect give LearnerBot minecraft:regeneration 5 1 true')
          learner.chat('/effect give LearnerBot minecraft:absorption 30 0 true')
        } catch {}
      } else {
        wastedConsumable = true
      }
      break
    case 14:
      if (lLoad.splashes > 0) {
        const hp = learner.health ?? 20
        if (hp >= 18.5) wastedConsumable = true
        lLoad.splashes--
        usedConsumable = 'instant_heal'
        try {
          learner.chat('/effect give LearnerBot minecraft:instant_health 1 1 true')
        } catch {}
      } else {
        wastedConsumable = true
      }
      break
    case 15:
      if (lLoad.pearls > 0 && now >= lLoad.pearlCdUntil) {
        lLoad.pearls--
        lLoad.pearlCdUntil = now + 1500
        usedConsumable = 'ender_pearl'
        try {
          if (equipItemByKeyword(learner, 'ender_pearl', 'hand')) {
            learner.activateItem()
          }
        } catch {}
        setTimeout(() => equipMainWeapon(learner, lLoad), 150)
      } else {
        wastedConsumable = true
      }
      break
    case 16:
      if (lLoad.buffs > 0 && now >= lLoad.buffUntil) {
        lLoad.buffs--
        lLoad.buffUntil = now + 15000
        usedConsumable = 'buff_potion'
        try {
          learner.chat('/effect give LearnerBot minecraft:speed 15 1 true')
          learner.chat('/effect give LearnerBot minecraft:strength 15 0 true')
        } catch {}
      } else {
        wastedConsumable = true
      }
      break
    case 17:
      if (lLoad.ranged > 0 && targetEntity) {
        lLoad.ranged--
        usedConsumable = lLoad.rangedSpec.id
        try {
          if (equipItemByKeyword(learner, lLoad.rangedSpec.item, 'hand')) {
            learner.activateItem()
          }
        } catch {}
        setTimeout(() => equipMainWeapon(learner, lLoad), 150)
      } else {
        wastedConsumable = true
      }
      break
    case 18:
      // Place Tactical Block Barricade / Pillar cover between LearnerBot and EnemyBot
      if (lLoad.blocks >= 2 && targetEntity) {
        lLoad.blocks -= 2
        obstacleCount = Math.min(6, obstacleCount + 1)
        placedBlock = lLoad.blockItem
        usedConsumable = 'block_barricade'
        placeBlockBetween(learner, targetEntity.position, lLoad.blockItem)
        learner.setControlState('back', true)
      } else {
        wastedConsumable = true
      }
      break
    case 19:
      // Place Cobweb or Lava Trap at enemy position
      if (lLoad.cobwebs > 0 && targetEntity) {
        const dist = learner.entity.position.distanceTo(targetEntity.position)
        if (dist <= 4.5) {
          lLoad.cobwebs--
          placedBlock = 'minecraft:cobweb'
          usedConsumable = 'cobweb_trap'
          const ex = Math.round(targetEntity.position.x)
          const ez = Math.round(targetEntity.position.z)
          try { learner.chat(`/setblock ${ex} 65 ${ez} minecraft:cobweb replace`) } catch {}
        } else {
          wastedConsumable = true
        }
      } else if (lLoad.lavaBuckets > 0 && targetEntity) {
        const dist = learner.entity.position.distanceTo(targetEntity.position)
        if (dist <= 4.2 && dist >= 1.8) {
          lLoad.lavaBuckets--
          placedBlock = 'minecraft:lava'
          usedConsumable = 'lava_trap'
          const ex = Math.round(targetEntity.position.x)
          const ez = Math.round(targetEntity.position.z)
          try {
            learner.chat(`/setblock ${ex} 65 ${ez} minecraft:lava replace`)
            setTimeout(() => {
              try { if (learner && learnerSpawned) learner.chat(`/setblock ${ex} 65 ${ez} minecraft:air replace`) } catch {}
            }, 1500)
          } catch {}
        } else {
          wastedConsumable = true
        }
      } else {
        wastedConsumable = true
      }
      break
    case 20:
      // Use Water Bucket (extinguish fire, clear cobweb/lava, push enemy back)
      if (lLoad.waterBuckets > 0) {
        lLoad.waterBuckets--
        usedConsumable = 'water_bucket'
        const lx = Math.round(learner.entity.position.x)
        const lz = Math.round(learner.entity.position.z)
        try {
          learner.chat(`/setblock ${lx} 65 ${lz} minecraft:water replace`)
          setTimeout(() => {
            try { if (learner && learnerSpawned) learner.chat(`/setblock ${lx} 65 ${lz} minecraft:air replace`) } catch {}
          }, 800)
        } catch {}
      } else {
        wastedConsumable = true
      }
      break
    case 21:
      // Pillar-up / High-ground block jump under feet
      if (lLoad.blocks > 0) {
        lLoad.blocks--
        placedBlock = lLoad.blockItem
        usedConsumable = 'pillar_jump'
        learner.setControlState('jump', true)
        const lx = Math.round(learner.entity.position.x)
        const lz = Math.round(learner.entity.position.z)
        setTimeout(() => {
          try { if (learner && learnerSpawned) learner.chat(`/setblock ${lx} 65 ${lz} ${lLoad.blockItem} replace`) } catch {}
        }, 120)
      } else {
        wastedConsumable = true
      }
      break
    case 22: {
      // END CRYSTAL: obsidian base + crystal at the enemy, detonated at 2.4-5.5 blocks
      if ((lLoad.crystals || 0) > 0 && (lLoad.obsidian || 0) > 0 && targetEntity) {
        const ex = Math.round(targetEntity.position.x)
        const ez = Math.round(targetEntity.position.z)
        const dist = learner.entity.position.distanceTo(targetEntity.position)
        if (dist >= 2.2 && dist <= 5.8) {
          lLoad.crystals -= 1
          lLoad.obsidian -= 1
          placedBlock = 'minecraft:end_crystal'
          usedConsumable = 'end_crystal'
          try {
            learner.chat(`/setblock ${ex} 65 ${ez} minecraft:obsidian replace`)
            learner.chat(`/setblock ${ex} 66 ${ez} minecraft:end_crystal replace`)
            learner.chat(`/arena explode ${ex} 66 ${ez} 10.5 24 ${learner.username}`)
          } catch {}
          learner.setControlState('back', true)
        } else {
          wastedConsumable = true
        }
      } else {
        wastedConsumable = true
      }
      break
    }
    case 23: {
      // RESPAWN ANCHOR charged with glowstone
      if ((lLoad.anchors || 0) > 0 && targetEntity) {
        const dist = learner.entity.position.distanceTo(targetEntity.position)
        if (dist >= 1.4 && dist <= 4.9) {
          lLoad.anchors -= 1
          placedBlock = 'minecraft:respawn_anchor'
          usedConsumable = 'respawn_anchor'
          const ex = Math.round(targetEntity.position.x)
          const ez = Math.round(targetEntity.position.z)
          try {
            learner.chat(`/setblock ${ex} 65 ${ez} minecraft:respawn_anchor replace`)
            learner.chat(`/arena explode ${ex} 65 ${ez} 9 20 ${learner.username}`)
          } catch {}
        } else {
          wastedConsumable = true
        }
      } else {
        wastedConsumable = true
      }
      break
    }
    case 24: {
      // TNT MINECART artillery
      if ((lLoad.minecarts || 0) > 0 && targetEntity) {
        const dist = learner.entity.position.distanceTo(targetEntity.position)
        if (dist >= 2.8 && dist <= 8.8) {
          lLoad.minecarts -= 1
          placedBlock = 'minecraft:tnt_minecart'
          usedConsumable = 'tnt_minecart'
          const ex = Math.round(targetEntity.position.x)
          const ez = Math.round(targetEntity.position.z)
          try {
            learner.chat(`/setblock ${ex} 65 ${ez} minecraft:rail replace`)
            learner.chat(`/setblock ${ex} 65 ${ez} minecraft:tnt_minecart replace`)
            learner.chat(`/arena explode ${ex} 65 ${ez} 7.5 17 ${learner.username}`)
          } catch {}
        } else {
          wastedConsumable = true
        }
      } else {
        wastedConsumable = true
      }
      break
    }
    case 25: {
      // OBSIDIAN BLAST CAGE around self
      if ((lLoad.obsidian || 0) >= 4) {
        lLoad.obsidian -= 4
        lLoad.cageUntil = now + 4000
        placedBlock = 'minecraft:obsidian'
        usedConsumable = 'obsidian_cage'
        try { learner.chat(`/arena cage ${learner.username} 4000`) } catch {}
      } else {
        wastedConsumable = true
      }
      break
    }
    case 26: {
      // W-TAP sprint-reset combo: hit, cancel knockback, keep full sprint pressure
      if (targetEntity) {
        const dist = learner.entity.position.distanceTo(targetEntity.position)
        if (dist <= 3.6) {
          usedConsumable = 'wtap_combo'
          try {
            learner.setControlState('sprint', true)
            learner.setControlState('forward', true)
            tryAttack()
            learner.chat(`/arena wtap ${learner.username}`)
          } catch {}
        } else {
          wastedConsumable = true
        }
      } else {
        wastedConsumable = true
      }
      break
    }
    case 27: {
      // PEARL + CRYSTAL blink combo
      if ((lLoad.pearls || 0) > 0 && (lLoad.crystals || 0) > 0 && (lLoad.obsidian || 0) > 0 && targetEntity) {
        const dist = learner.entity.position.distanceTo(targetEntity.position)
        if (dist >= 3.4 && dist <= 12.5) {
          lLoad.pearls -= 1
          lLoad.crystals -= 1
          lLoad.obsidian -= 1
          placedBlock = 'minecraft:end_crystal'
          usedConsumable = 'pearl_crystal'
          const ex = Math.round(targetEntity.position.x)
          const ez = Math.round(targetEntity.position.z)
          try {
            if (equipItemByKeyword(learner, 'ender_pearl', 'hand')) learner.activateItem()
            learner.chat(`/setblock ${ex} 65 ${ez} minecraft:obsidian replace`)
            learner.chat(`/setblock ${ex} 66 ${ez} minecraft:end_crystal replace`)
            setTimeout(() => {
              try { if (learner && learnerSpawned) learner.chat(`/arena explode ${ex} 66 ${ez} 10.5 24 ${learner.username}`) } catch {}
            }, 250)
          } catch {}
          setTimeout(() => equipMainWeapon(learner, lLoad), 400)
        } else {
          wastedConsumable = true
        }
      } else {
        wastedConsumable = true
      }
      break
    }
    default: break
  }

  return { usedConsumable, wastedConsumable, brokeEnemyShield, placedBlock }
}

// ---------------------------------------------------------------------------
// PRO BOT AI — plays like a tournament 1.16.5 PvP player:
// crystal/anchor/TNT-cart combos, shield-timing, axe shield-breaks, kiting,
// obsidian cages, pearls, traps, goldens and W-tap combos.
// ---------------------------------------------------------------------------
let eUtilityCdUntil = 0
let eComboCdUntil = 0

function enemySay(cmd) {
  try { if (learner && learnerSpawned) learner.chat(cmd) } catch {}
}

function enemyDiag(cmd) {
  try { if (learner && learnerSpawned) learner.chat(cmd) } catch {}
}

function enemyExplode(x, z, y, power, selfPower) {
  enemySay(`/arena explode ${Math.round(x)} ${y} ${Math.round(z)} ${power} ${selfPower} ${enemy.username}`)
}

function enemyCage() {
  enemySay(`/arena cage ${enemy.username} 4000`)
  eLoad.cageUntil = Date.now() + 4000
}

const enemyTimer = setInterval(() => {
  if (!botsReady() || episodeDone || resetting) return
  const now = Date.now()
  const targetEntity = getPeerEntity(enemy, learner)
  if (!targetEntity) return
  const dist = enemy.entity.position.distanceTo(targetEntity.position)
  safeLookAt(enemy, targetEntity.position.offset(0, 1.5, 0))
  clearControls(enemy)

  const eHP = enemy.health ?? 20
  const lHP = learner.health ?? 20

  const enemySwing = () => {
    if (!targetEntity) return
    if (now - lastEnemyAttackTime < eLoad.weapon.cdMs) return
    try {
      equipMainWeapon(enemy, eLoad)
      enemy.attack(targetEntity)
      lastEnemyAttackTime = now
      if (eLoad.weapon.shieldBreak && lLoad.shieldUp && now >= lLoad.shieldDisabledUntil) {
        lLoad.shieldUp = false
        lLoad.shieldDisabledUntil = now + 5000
        try { learner.deactivateItem() } catch {}
      }
    } catch {}
  }

  // 1. defensive utility: water out of hazards / heal / cage against explosives
  if ((eLoad.hazard || 0) > 0) eLoad.hazard = 0
  if (eHP <= 12 && eLoad.splashes > 0) {
    eLoad.splashes -= 1
    enemySay('/effect give EnemyBot minecraft:instant_health 1 1 true')
    return
  }
  if (eHP <= 14 && eLoad.gaps > 0 && now >= eLoad.regenUntil) {
    eLoad.gaps -= 1
    eLoad.regenUntil = now + 5000
    enemySay('/effect give EnemyBot minecraft:regeneration 5 1 true')
    enemySay('/effect give EnemyBot minecraft:absorption 30 0 true')
    return
  }
  if (eHP <= 13 && (eLoad.obsidian || 0) >= 4 && (lLoad.crystals + lLoad.anchors + lLoad.minecarts) > 0
      && (lLoad.crystals || 0) > 0 && Math.random() < 0.4) {
    eLoad.obsidian -= 4
    enemyCage()
    return
  }

  // 2. PRO explosive combo tier (shared cooldown so it can't chain every tick)
  if (now >= eUtilityCdUntil && dist >= 2.2) {
    if ((eLoad.crystals || 0) > 0 && (eLoad.obsidian || 0) > 0 && dist >= 2.4 && dist <= 5.4 && lHP <= 18 && Math.random() < 0.55) {
      eLoad.crystals -= 1
      eLoad.obsidian -= 1
      eUtilityCdUntil = now + 2100
      const ex = targetEntity.position.x
      const ez = targetEntity.position.z
      enemySay(`/setblock ${Math.round(ex)} 65 ${Math.round(ez)} minecraft:obsidian replace`)
      enemySay(`/setblock ${Math.round(ex)} 66 ${Math.round(ez)} minecraft:end_crystal replace`)
      setTimeout(() => { try { enemyExplode(ex, ez, 66, 10.5, 24) } catch {} }, 120)
      enemy.setControlState('back', true)
      return
    }
    if ((eLoad.anchors || 0) > 0 && dist >= 1.6 && dist <= 4.6 && Math.random() < 0.5) {
      eLoad.anchors -= 1
      eUtilityCdUntil = now + 1900
      const ex = targetEntity.position.x
      const ez = targetEntity.position.z
      enemySay(`/setblock ${Math.round(ex)} 65 ${Math.round(ez)} minecraft:respawn_anchor replace`)
      setTimeout(() => { try { enemyExplode(ex, ez, 65, 9, 20) } catch {} }, 120)
      return
    }
    if ((eLoad.minecarts || 0) > 0 && dist >= 3.0 && dist <= 8.6 && Math.random() < 0.45) {
      eLoad.minecarts -= 1
      eUtilityCdUntil = now + 1900
      const ex = targetEntity.position.x
      const ez = targetEntity.position.z
      enemySay(`/setblock ${Math.round(ex)} 65 ${Math.round(ez)} minecraft:rail replace`)
      setTimeout(() => { try { enemyExplode(ex, ez, 65, 7.5, 17) } catch {} }, 120)
      return
    }
  }

  // 3. axe shield-break opening
  if (lLoad.hasShield && lLoad.shieldUp && now >= lLoad.shieldDisabledUntil && dist <= 3.4
      && eLoad.weapon.shieldBreak && now - lastEnemyAttackTime >= eLoad.weapon.cdMs) {
    enemySwing()
    return
  }

  // 4. utility: barricade, traps, buff, pearl-in
  if (eHP <= 14 && eLoad.blocks >= 2 && dist < 4.8 && Math.random() < 0.3) {
    eLoad.blocks -= 2
    obstacleCount = Math.min(6, obstacleCount + 1)
    placeBlockBetween(enemy, targetEntity.position, eLoad.blockItem)
    enemy.setControlState('back', true)
    return
  }
  if (eLoad.cobwebs > 0 && dist <= 4.0 && Math.random() < 0.2) {
    eLoad.cobwebs -= 1
    enemySay(`/setblock ${Math.round(targetEntity.position.x)} 65 ${Math.round(targetEntity.position.z)} minecraft:cobweb replace`)
    return
  }
  if ((eLoad.lavaBuckets || 0) > 0 && dist >= 2.0 && dist <= 4.2 && Math.random() < 0.2) {
    eLoad.lavaBuckets -= 1
    const ex = Math.round(targetEntity.position.x)
    const ez = Math.round(targetEntity.position.z)
    enemySay(`/setblock ${ex} 65 ${ez} minecraft:lava replace`)
    setTimeout(() => { try { enemySay(`/setblock ${ex} 65 ${ez} minecraft:air replace`) } catch {} }, 1500)
    return
  }
  if (eLoad.buffs > 0 && now >= eLoad.buffUntil && dist > 4.0 && Math.random() < 0.3) {
    eLoad.buffs -= 1
    eLoad.buffUntil = now + 15000
    enemySay('/effect give EnemyBot minecraft:speed 15 1 true')
    enemySay('/effect give EnemyBot minecraft:strength 15 0 true')
    return
  }
  if (eLoad.pearls > 0 && now >= eLoad.pearlCdUntil && dist > 7.0 && eHP > 12 && Math.random() < 0.3) {
    eLoad.pearls -= 1
    eLoad.pearlCdUntil = now + 2500
    try {
      if (equipItemByKeyword(enemy, 'ender_pearl', 'hand')) enemy.activateItem()
    } catch {}
    setTimeout(() => equipMainWeapon(enemy, eLoad), 200)
    return
  }

  // 5. shield timing (only while the learner's swing is nearly ready)
  const learnerArmingUp = now - lastAttackTime > lLoad.weapon.cdMs * 0.55
  if (eLoad.hasShield && now >= eLoad.shieldDisabledUntil && dist < 3.5 && learnerArmingUp && Math.random() < 0.55) {
    eLoad.shieldUp = true
    try { enemy.activateItem(true) } catch {}
  } else if (eLoad.shieldUp) {
    eLoad.shieldUp = false
    try { enemy.deactivateItem() } catch {}
  }

  // 6. W-tap pressure & melee
  if (dist > 2.6) {
    enemy.setControlState('forward', true)
    if (dist > 4.5) enemy.setControlState('sprint', true)
  } else if (dist < 2.0) {
    enemy.setControlState('back', true)
  }
  if (!eLoad.shieldUp && dist < 3.2) {
    if (now >= eComboCdUntil && Math.random() < 0.35) {
      eComboCdUntil = now + 900
      enemy.setControlState('sprint', true)
      enemySwing()
      try { enemySay(`/arena wtap ${enemy.username}`) } catch {}
    } else if (Math.random() < 0.8) {
      enemySwing()
    }
  }
  if (Math.random() < 0.18) enemy.setControlState('left', true)
  if (Math.random() < 0.18) enemy.setControlState('right', true)
  if (dist < 4.0 && Math.random() < 0.12) enemy.setControlState('jump', true)
}, 150)

const wss = new WebSocketServer({ port: WS_PORT })
console.log(`WebSocket server on ${WS_PORT} - waiting for Python...`)

wss.on('connection', (ws) => {
  console.log('Python connected')
  ws.on('message', (data) => {
    let msg
    try { msg = JSON.parse(data.toString()) } catch { return }
    if (msg.ping) {
      try {
        ws.send(JSON.stringify({ pong: true, ready: botsReady() }))
      } catch {}
      return
    }
    if (msg.reset) {
      const started = resetEpisode()
      setTimeout(() => {
        try {
          ws.send(JSON.stringify({
            obs: getObservation(),
            info: {
              ready: Boolean(started && botsReady()),
              learner_kit: lLoad.summary,
              enemy_kit: eLoad.summary,
            },
            done: false,
          }))
        } catch (e) { console.log('WebSocket reset reply error:', e.message) }
      }, started ? 2100 : 500)
      return
    }
    if (!botsReady()) {
      try {
        ws.send(JSON.stringify({
          obs: getObservation(),
          done: true,
          info: {
            ready: false,
            damage_dealt: 0,
            damage_taken: 0,
            healed: 0,
            learner_dead: false,
            enemy_dead: false,
            timeout: true,
            bot_disconnected: true,
          },
        }))
      } catch {}
      return
    }
    stepCount++
    const actionRes = doAction(Number(msg.action) || 0)
    setTimeout(() => {
      const learnerHP = learner && learner.health !== undefined ? learner.health : 0
      const enemyHP = enemy && enemy.health !== undefined ? enemy.health : 0
      const warmupNow = Date.now() - episodeLiveAt < 1500 || stepCount <= 2
      const damageDealt = warmupNow ? 0 : Math.max(0, prevEnemyHP - enemyHP)
      const damageTaken = warmupNow ? 0 : Math.max(0, prevLearnerHP - learnerHP)
      const healed = Math.max(0, learnerHP - prevLearnerHP)
      prevLearnerHP = learnerHP
      prevEnemyHP = enemyHP
      const warmup = Date.now() - episodeLiveAt < 1500 || stepCount <= 2
      const learnerDead = !warmup && learnerHP > 0 && learnerHP <= 6
      const enemyDead = !warmup && enemyHP > 0 && enemyHP <= 6
      const timeout = stepCount >= MAX_STEPS
      const done = (!warmup && (learnerDead || enemyDead)) || timeout || !botsReady()
      if (done) {
        episodeDone = true
        clearControls(learner)
        clearControls(enemy)
      }
      const lp = learner && learner.entity ? learner.entity.position : null
      const ep = enemy && enemy.entity ? enemy.entity.position : null
      const dist = lp && ep ? lp.distanceTo(ep) : 0
      try {
        ws.send(JSON.stringify({
          obs: getObservation(),
          done,
          info: {
            ready: botsReady(),
            learner_kit: lLoad.summary,
            enemy_kit: eLoad.summary,
            damage_dealt: damageDealt,
            damage_taken: damageTaken,
            healed,
            trapped_enemy: actionRes.usedConsumable === 'cobweb_trap' || actionRes.usedConsumable === 'lava_trap',
            crystal_hit: actionRes.usedConsumable === 'end_crystal' || actionRes.usedConsumable === 'pearl_crystal',
            anchor_hit: actionRes.usedConsumable === 'respawn_anchor',
            minecart_hit: actionRes.usedConsumable === 'tnt_minecart',
            cage_blocked: actionRes.usedConsumable === 'obsidian_cage',
            wtap_combo: actionRes.usedConsumable === 'wtap_combo',
            tactical_cover: actionRes.usedConsumable === 'block_barricade' && learnerHP <= 15,
            used_consumable: actionRes.usedConsumable,
            placed_block: actionRes.placedBlock,
            wasted_consumable: actionRes.wastedConsumable,
            shield_broken: actionRes.brokeEnemyShield,
            learner_hp: learnerHP,
            enemy_hp: enemyHP,
            distance: dist,
            learner_dead: learnerDead,
            enemy_dead: enemyDead,
            timeout,
          },
        }))
      } catch {}
    }, STEP_DELAY_MS)
  })
})

function setupArena() {
  if (!learner || !learnerSpawned) return
  clearControls(learner)
  const cmds = [
    '/fill -11 65 -11 11 80 11 minecraft:air',
    '/fill -10 64 -10 10 64 10 minecraft:stone',
    '/fill -10 65 -10 -10 72 10 minecraft:bedrock',
    '/fill 10 65 -10 10 72 10 minecraft:bedrock',
    '/fill -10 65 -10 10 72 -10 minecraft:bedrock',
    '/fill -10 65 10 10 72 10 minecraft:bedrock',
    '/setworldspawn 0 65 0',
    '/gamerule doMobSpawning false',
    '/gamerule doDaylightCycle false',
    '/gamerule doWeatherCycle false',
    '/gamerule keepInventory true',
    '/gamerule doFireTick false',
    '/time set day',
    '/weather clear',
  ]
  cmds.forEach((c, i) => setTimeout(() => {
    try { if (learner && learnerSpawned) learner.chat(c) } catch {}
  }, 350 * (i + 1)))
  console.log('Arena setup commands sent')
}

function createLearner() {
  if (learnerReconnecting) return
  console.log(`Connecting LearnerBot to ${HOST}:${PORT}...`)
  learnerSpawned = false
  learner = mineflayer.createBot({ host: HOST, port: PORT, username: 'LearnerBot', version: MC_VERSION })
  installMovementGuard(learner, 'LearnerBot')
  learner.once('spawn', () => {
    learnerSpawned = true
    console.log('LearnerBot spawned')
    if (!arenaDone) {
      setTimeout(setupArena, 1500)
      setTimeout(resetEpisode, 6500)
      arenaDone = true
    } else {
      setTimeout(resetEpisode, 1000)
    }
  })
  learner.on('death', () => {
    episodeDone = true
    learner.physicsEnabled = false
    try { learner.respawn() } catch {}
  })
  learner.on('spawn', () => {
    learnerSpawned = true
    learner.physicsEnabled = true
  })
  learner.on('kicked', (r) => console.log('LearnerBot kicked:', r))
  learner.on('error', (e) => console.log('LearnerBot error:', e.message))
  learner.on('end', () => {
    learnerSpawned = false
    console.log('LearnerBot disconnected. Reconnecting in 5 sec...')
    learnerReconnecting = true
    setTimeout(() => { learnerReconnecting = false; createLearner() }, 5000)
  })
}

function createEnemy() {
  if (enemyReconnecting) return
  console.log(`Connecting EnemyBot to ${HOST}:${PORT}...`)
  enemySpawned = false
  enemy = mineflayer.createBot({ host: HOST, port: PORT, username: 'EnemyBot', version: MC_VERSION })
  installMovementGuard(enemy, 'EnemyBot')
  enemy.once('spawn', () => {
    enemySpawned = true
    console.log('EnemyBot spawned')
  })
  enemy.on('death', () => {
    episodeDone = true
    enemy.physicsEnabled = false
    try { enemy.respawn() } catch {}
  })
  enemy.on('spawn', () => {
    enemySpawned = true
    enemy.physicsEnabled = true
  })
  enemy.on('kicked', (r) => console.log('EnemyBot kicked:', r))
  enemy.on('error', (e) => console.log('EnemyBot error:', e.message))
  enemy.on('end', () => {
    enemySpawned = false
    console.log('EnemyBot disconnected. Reconnecting in 7 sec...')
    enemyReconnecting = true
    setTimeout(() => { enemyReconnecting = false; createEnemy() }, 7000)
  })
}

function shutdown() {
  clearInterval(enemyTimer)
  try { wss.close() } catch {}
  try { if (learner) learner.quit() } catch {}
  try { if (enemy) enemy.quit() } catch {}
  process.exit(0)
}

process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)

createLearner()
setTimeout(createEnemy, 2500)

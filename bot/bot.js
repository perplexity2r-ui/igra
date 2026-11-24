// Minecraft PvP AI — bots on Mineflayer (1.16.5)
// LearnerBot — controlled by neural net from Python via WebSocket (port 8765)
// EnemyBot   — simple opponent: walks to LearnerBot and attacks
// Each bot reconnects separately (no duplicate bots!)

const mineflayer = require('mineflayer')
const { WebSocketServer } = require('ws')

const HOST = 'localhost'
const PORT = 25565
const WS_PORT = 8765

let learner = null
let enemy = null
let learnerReconnecting = false
let enemyReconnecting = false
let lastAttackTime = 0
let prevLearnerHP = 20
let prevEnemyHP = 20
let episodeDone = false
let stepCount = 0
const MAX_STEPS = 600
let arenaDone = false

function clearControls(bot) {
  if (!bot) return
  for (const c of ['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak']) {
    try { bot.setControlState(c, false) } catch {}
  }
}

function resetEpisode() {
  if (!learner || !enemy || !learner.entity || !enemy.entity) return false
  clearControls(learner)
  clearControls(enemy)
  episodeDone = true

  // Keep both bots alive. Old Paper 1.16.5 can reject Mineflayer movement
  // immediately after forced /tp or /kill. Healing is stable and sufficient
  // because arena walls keep both players inside the episode area.
  const cmds = [
    '/clear LearnerBot',
    '/clear EnemyBot',
    '/give LearnerBot minecraft:diamond_sword 1',
    '/give EnemyBot minecraft:diamond_sword 1',
    '/effect give LearnerBot minecraft:instant_health 1 10 true',
    '/effect give EnemyBot minecraft:instant_health 1 10 true',
    '/effect give LearnerBot minecraft:saturation 1 10 true',
    '/effect give EnemyBot minecraft:saturation 1 10 true',
  ]
  cmds.forEach((cmd, i) => setTimeout(() => {
    try { learner.chat(cmd) } catch {}
  }, i * 120))

  setTimeout(() => {
    prevLearnerHP = learner && learner.health !== undefined ? learner.health : 20
    prevEnemyHP = enemy && enemy.health !== undefined ? enemy.health : 20
    episodeDone = false
    stepCount = 0
  }, 1200)
  return true
}

function getObservation() {
  if (!learner || !enemy || !learner.entity || !enemy.entity) {
    return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  }
  const lp = learner.entity.position
  const ep = enemy.entity.position
  const dx = ep.x - lp.x
  const dz = ep.z - lp.z
  const dist = Math.sqrt(dx * dx + dz * dz)
  const cooldownReady = (Date.now() - lastAttackTime > 500) ? 1 : 0
  return [
    (learner.health || 0) / 20,
    (enemy.health || 0) / 20,
    Math.min(dist / 20, 1),
    learner.entity.velocity.length(),
    enemy.entity.velocity.length(),
    dx / 20,
    dz / 20,
    learner.entity.onGround ? 1 : 0,
    cooldownReady,
    stepCount / MAX_STEPS,
  ]
}

function doAction(action) {
  if (!learner || !enemy || !learner.entity || !enemy.entity) return
  clearControls(learner)
  const targetPos = enemy.entity.position.offset(0, 1.5, 0)
  learner.lookAt(targetPos, true).catch(() => {})
  const tryAttack = () => {
    if (Date.now() - lastAttackTime > 400) {
      learner.attack(enemy.entity)
      lastAttackTime = Date.now()
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
    case 9: learner.setControlState('forward', true); learner.setControlState('jump', true); learner.setControlState('sprint', true); break
    case 10: learner.activateItem(); setTimeout(() => { try { learner.deactivateItem() } catch {} }, 300); break
    case 11: learner.setControlState('back', true); tryAttack(); break
    default: break
  }
}

setInterval(() => {
  if (!enemy || !learner || !enemy.entity || !learner.entity || episodeDone) return
  const dist = enemy.entity.position.distanceTo(learner.entity.position)
  enemy.lookAt(learner.entity.position.offset(0, 1.5, 0), true).catch(() => {})
  clearControls(enemy)
  if (dist > 2.5) enemy.setControlState('forward', true)
  if (dist < 3 && Math.random() < 0.7) enemy.attack(learner.entity)
  if (Math.random() < 0.15) enemy.setControlState('left', true)
  if (Math.random() < 0.15) enemy.setControlState('right', true)
  if (Math.random() < 0.1) enemy.setControlState('jump', true)
}, 150)

const wss = new WebSocketServer({ port: WS_PORT })
console.log(`WebSocket server on ${WS_PORT} - waiting for Python...`)

wss.on('connection', (ws) => {
  console.log('Python connected')
  ws.on('message', (data) => {
    let msg
    try { msg = JSON.parse(data.toString()) } catch { return }
    if (msg.reset) {
      const started = resetEpisode()
      setTimeout(() => {
        try {
          ws.send(JSON.stringify({
            obs: getObservation(),
            info: { ready: Boolean(started) },
            done: false,
          }))
        } catch (e) { console.log('WebSocket reset reply error:', e.message) }
      }, started ? 1400 : 500)
      return
    }
    stepCount++
    doAction(msg.action)
    setTimeout(() => {
      const learnerHP = learner && learner.health !== undefined ? learner.health : 0
      const enemyHP = enemy && enemy.health !== undefined ? enemy.health : 0
      const damageDealt = Math.max(0, prevEnemyHP - enemyHP)
      const damageTaken = Math.max(0, prevLearnerHP - learnerHP)
      prevLearnerHP = learnerHP
      prevEnemyHP = enemyHP
      // End before actual death; reset heals both bots without respawn glitches.
      const learnerDead = learnerHP <= 8
      const enemyDead = enemyHP <= 8
      const timeout = stepCount >= MAX_STEPS
      const done = learnerDead || enemyDead || timeout
      try {
        ws.send(JSON.stringify({
          obs: getObservation(),
          done,
          info: { damage_dealt: damageDealt, damage_taken: damageTaken, learner_dead: learnerDead, enemy_dead: enemyDead, timeout },
        }))
      } catch {}
    }, 100)
  })
})

function setupArena() {
  if (!learner) return
  clearControls(learner)
  const cmds = [
    // Clear terrain first: an existing world may not be flat.
    '/fill -11 65 -11 11 80 11 minecraft:air',
    '/fill -10 64 -10 10 64 10 minecraft:stone',
    '/fill -10 65 -10 -10 70 10 minecraft:bedrock',
    '/fill 10 65 -10 10 70 10 minecraft:bedrock',
    '/fill -10 65 -10 10 70 -10 minecraft:bedrock',
    '/fill -10 65 10 10 70 10 minecraft:bedrock',
    '/gamerule doMobSpawning false',
    '/time set day',
    '/weather clear',
  ]
  cmds.forEach((c, i) => setTimeout(() => { try { learner.chat(c) } catch {} }, 500 * (i + 1)))
  console.log('Arena setup commands sent')
}

function createLearner() {
  if (learnerReconnecting) return
  console.log('Connecting LearnerBot...')
  learner = mineflayer.createBot({ host: HOST, port: PORT, username: 'LearnerBot', version: '1.16.5' })
  learner.once('spawn', () => {
    console.log('LearnerBot spawned')
    if (!arenaDone) {
      setTimeout(setupArena, 2000)
      setTimeout(resetEpisode, 6000)
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
  learner.on('spawn', () => { learner.physicsEnabled = true })
  learner.on('kicked', (r) => console.log('LearnerBot kicked:', r))
  learner.on('error', (e) => console.log('LearnerBot error:', e.message))
  learner.on('end', () => {
    console.log('LearnerBot disconnected. Reconnecting in 10 sec...')
    learnerReconnecting = true
    setTimeout(() => { learnerReconnecting = false; createLearner() }, 10000)
  })
}

function createEnemy() {
  if (enemyReconnecting) return
  console.log('Connecting EnemyBot...')
  enemy = mineflayer.createBot({ host: HOST, port: PORT, username: 'EnemyBot', version: '1.16.5' })
  enemy.once('spawn', () => console.log('EnemyBot spawned'))
  enemy.on('death', () => {
    episodeDone = true
    enemy.physicsEnabled = false
    try { enemy.respawn() } catch {}
  })
  enemy.on('spawn', () => { enemy.physicsEnabled = true })
  enemy.on('kicked', (r) => console.log('EnemyBot kicked:', r))
  enemy.on('error', (e) => console.log('EnemyBot error:', e.message))
  enemy.on('end', () => {
    console.log('EnemyBot disconnected. Reconnecting in 15 sec...')
    enemyReconnecting = true
    // Keep this different from LearnerBot's delay to avoid Paper IP throttling.
    setTimeout(() => { enemyReconnecting = false; createEnemy() }, 15000)
  })
}

createLearner()
// Paper throttles rapid connections from the same IP. Start the second bot later.
setTimeout(createEnemy, 5000)

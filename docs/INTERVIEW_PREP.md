# TowerLogic interview prep

This is a factual story bank for discussing TowerLogic as Ethan B. Chen. Keep
the wording conversational and adapt the detail to the role. The project is an
experimental CV/ML automation system; it should not be presented as a
production-grade or universally reliable game-playing agent.

## Short pitches

### 30 seconds

“TowerLogic is a Python computer-vision and ML project I built around Clash
Royale. It captures emulator screenshots, recognizes the hand and battlefield,
turns those observations into a normalized game-state vector, scores candidate
card placements with a small PyTorch policy plus safety rules, and can send the
result through ADB. I recently rehabilitated it on Apple Silicon and BlueStacks
Air, adding modern foreground detection, reward-screen recovery, and a dry-run
mode that logs intended actions without sending taps.”

### 60 seconds

“The interesting part of TowerLogic is the boundary between perception and
control. OpenCV handles screenshots, fixed UI checks, and templates. A small
PyTorch classifier identifies supported hand cards, and a YOLO-style detector
exported to ONNX finds field objects. The bot combines those results with
elixir, timing, lane, tower-health, and crown features. It generates legal
candidate actions, scores them with a 30-input policy network, applies explicit
placement and elixir masks, and then calls one emulator input layer. When I
revisited the project on a Mac with BlueStacks Air, several old assumptions
failed: the metadata file was gone, Android foreground output had changed, UI
pixel signatures were stale, and reward screens interrupted startup. I fixed
those at their boundaries and added dry-run tests so I could validate the whole
pipeline safely.”

## Architecture walkthrough

1. The GUI builds a job dictionary and starts `WorkerProcess` in a separate
   process.
2. The selected emulator controller connects to ADB, configures the validated
   419×633 framebuffer, and launches Clash Royale with its resolved launcher
   activity.
3. Navigation checks use screenshots plus Android activity/window state to
   distinguish startup, menu, battle, and reward interruptions.
4. During a battle, card crops go through classifier/template recognition and
   the arena crop goes through ONNX detection. These are best-effort visual
   signals, not perfect truth.
5. `fight.py` builds a 26-value normalized state vector. Each candidate appends
   card index, group id, and normalized coordinates, giving the policy 30
   inputs.
6. `EpsilonGreedyPolicy` chooses among candidate actions. Rule-based masks still
   enforce elixir, lane, defensive, spell, and arena-bound constraints.
7. The controller sends a card tap followed by a placement tap, or logs both
   when `TOWERLOGIC_DRY_RUN=1`.
8. `ProcessLogger` publishes progress through a multiprocessing queue, and the
   GUI renders status, reward telemetry, and statistics.

## Story bank

### 1. Rehabilitating an older project

**Situation/problem:** I returned to a project that depended on an older
macOS/BlueStacks/Clash Royale environment. It imported, but the real runtime
had drifted.

**Investigation:** I mapped the complete path from process startup through ADB,
screenshots, navigation, recognition, policy, and input. I compared the code
with live BlueStacks output instead of assuming the README was current.

**Action:** I made small boundary fixes: package-relative model paths,
platform-aware BlueStacks discovery, modern ADB parsing, current UI detection,
bounded reward recovery, and centralized dry-run input suppression. I added
focused tests before touching gameplay behavior.

**Result/learning:** The project became reproducible enough for a new Mac
demo, while the architecture remained recognizable. I learned that legacy
automation projects usually fail at interfaces—paths, process state, pixels,
and device protocols—before they fail in the model itself.

**Likely follow-ups:** How did you separate a real regression from an
environment problem? Which changes were deliberately not made? How did you
prove that dry-run was safe?

### 2. BlueStacks Air compatibility

**Situation/problem:** BlueStacks Air did not expose the legacy
`MimMetaData.json` that the original controller expected.

**Investigation:** I inspected `bluestacks.conf`, the installed app bundle, the
instance’s ADB port, and live `hd-adb` output. The configuration itself had the
instance display name, internal name, framebuffer, DPI, and ADB port.

**Action:** The controller now treats legacy metadata as optional, resolves the
instance and port from current configuration, supports path overrides, and
uses the bundled ADB server on port 5041.

**Result/learning:** BlueStacks Air connected as `127.0.0.1:<port>` without
creating fake metadata. I kept the legacy lookup as a fallback because that
preserves older installations.

**Likely follow-ups:** What happens if the config is missing? Why not scan every
port? How do you avoid controlling the wrong instance?

### 3. Renderer and display mismatch

**Situation/problem:** A black surface can look like a model failure even when
the app process is alive.

**Investigation:** I separated process state, foreground activity, screenshot
availability, rendered-pixel validity, and visual recognition. I also compared
the controller’s 419×633 coordinate space with BlueStacks’ current settings.

**Action:** The controller enforces the expected framebuffer and configured
renderer, then validates screenshot dimensions and rendered pixels. The README
now distinguishes the 419×633 visual space from the BlueStacks Air controller’s
320 DPI default.

**Result/learning:** This made startup diagnostics actionable: a black frame is
reported as an emulator/renderer issue rather than silently blamed on CV.

**Likely follow-ups:** Why are coordinates fixed? What would you change for a
different viewport? What is automatic versus user configuration?

### 4. Foreground detection on Android 13

**Situation/problem:** Clash Royale was visibly in front and its process
existed, but the parser returned `None`.

**Investigation:** The old implementation relied on a single legacy
`mCurrentFocus`/`mResumedActivity` line. Android 13/BlueStacks exposed
`topResumedActivity` in activity dumpsys output.

**Action:** I query several modern and legacy sources and parse package/activity
components from the first reliable marker, with diagnostic logging when none
resolve. Fixtures cover modern and legacy formats.

**Result/learning:** Foreground detection no longer fails just because one old
field is absent. The key lesson was to make platform parsers layered and
observable instead of trusting one string format.

**Likely follow-ups:** Which source wins? What if output is contradictory? How
would you test this without an emulator?

### 5. Stale main-menu recognition

**Situation/problem:** The game reached a stable current main menu, but a
seven-pixel legacy signature returned false.

**Investigation:** I logged each sampled pixel and compared the stable screenshot
with both legacy templates. The repeated mismatch was a UI drift, not a failed
launch.

**Action:** I preserved the legacy signature and added a current detector using
two independent static cues: a trophy-road template and the battle-button
region. Diagnostics identify which detector matched.

**Result/learning:** Startup advanced without weakening the old threshold into
a likely false positive. Fixed-pixel checks are useful as cheap fallbacks, but
they need an explicit compatibility boundary.

**Likely follow-ups:** How did you avoid matching loading screens? How would you
collect new templates? What is the remaining maintenance risk?

### 6. Trophy/reward state recovery

**Situation/problem:** A modern trophy-box reward screen could appear before
the main menu or during a battle. Waiting for the normal state forever was not
recovery.

**Investigation:** I added a bounded detector for the screen’s broad color/layout
signature, then traced both startup and fight-loop call sites.

**Action:** Both paths reuse one recovery function. It repeatedly attempts the
known center action, re-screens after each attempt, and returns `battle`,
`main_menu`, `timeout`, or `not_detected`. Dry-run logs the intended click and
does not claim the screen changed.

**Result/learning:** The state machine can move through an intermediate screen
without relaunching the game. This is a good example of modeling recovery as a
state transition rather than sprinkling one-off clicks through callers.

**Likely follow-ups:** What if the visual signature changes? Why is the timeout
bounded? What does dry-run mean when no input can change the screenshot?

### 7. Safe automation with dry-run

**Situation/problem:** I needed to validate recognition and policy decisions
without accidentally playing a live match.

**Investigation:** I searched every tap/swipe path, including controller
overrides and raw `shell input` calls.

**Action:** A single environment flag, `TOWERLOGIC_DRY_RUN=1`, is checked at the
shared ADB controller and the MEmu override. Intended coordinates and durations
are logged; screenshots, queries, and app launch remain available. Unit tests
assert both suppressed and normal command behavior.

**Result/learning:** The full pipeline can be exercised with no gameplay input.
The boundary is explicit: app launch and display configuration are not
suppressed, so a dry run still needs a carefully scoped emulator.

**Likely follow-ups:** How do you know there is no bypass? What about a new
controller implementation? Would you make dry-run the default?

### 8. ML preprocessing bug

**Situation/problem:** OpenCV crops are BGR while PIL/model training expects
RGB. The runtime was passing the array directly to PIL.

**Investigation:** I evaluated the committed validation images two ways. On
the current 257-image validation set, the old direct-BGR path classified
252/257, while explicit BGR→RGB conversion classified 257/257.

**Action:** I added one conversion helper used by both classifier prediction
methods and a regression test for channel order.

**Result/learning:** This was a concrete, small bug with measurable impact. It
reinforced that “the model loaded” is not enough; preprocessing must match the
training contract.

**Likely follow-ups:** Why did the old path still score well? How do you prevent
RGB/BGR regressions? What is the difference between test-set accuracy and live
accuracy?

### 9. ONNX detector boundary handling

**Situation/problem:** YOLO export coordinates can extend outside an ROI or
source frame, and NMS expects `xywh` boxes rather than `xyxy`.

**Investigation:** I traced preprocessing, transpose, center-`xywh` decoding,
scaling, NMS, and drawing. I created a synthetic output that extended beyond a
100×100 source image.

**Action:** The runtime keeps the corrected `xywh` NMS conversion and clamps
decoded boxes to valid source bounds, dropping degenerate boxes. A regression
test protects that contract.

**Result/learning:** Field detections are safe for downstream arena and drawing
logic. The model’s confidence is still not a substitute for live accuracy.

**Likely follow-ups:** Why stretch instead of letterbox? How are classes mapped?
How would you evaluate detector precision/recall?

### 10. Learning graph lifecycle

**Situation/problem:** The graph could appear empty after a worker completed,
even though the worker had recorded metrics.

**Investigation:** I traced `OnlineTrainer.update()` to `Logger`, the
multiprocessing queue, `BotApplication._poll`, and `TowerLogicUI`. The completed
worker path replaced the main logger with a new empty logger.

**Action:** I kept the existing logger and its metrics when setting the UI back
to idle; the next start already creates a fresh logger. I also publish episode
metrics immediately through `ProcessLogger`.

**Result/learning:** The final run remains inspectable without changing
training. The graph is correctly described as rolling in-memory reward
telemetry, not a persisted loss dashboard.

**Likely follow-ups:** What exactly does each series mean? How would you persist
experiments? Why can a queue still lose updates?

## Common answers

### What did you personally build?

“I built the runtime around the emulator boundary, the visual recognition and
policy plumbing, the state-machine transitions, and the GUI configuration and
monitoring path. I also wrote the compatibility and regression work that made
the old project run again on Apple Silicon. The project uses a mix of learned
models and explicit rules, so I would not claim that the policy alone controls
everything.”

### What was the hardest part?

“The hardest part was deciding whether a failure belonged to the emulator, the
parser, the visual detector, or the policy. A live process with a black frame,
or a visible game with a missing `topResumedActivity`, can look like an ML bug.
I learned to log and test each boundary independently before changing behavior.”

### How does the ML work?

“There are two learned pieces in the runtime. A small PyTorch classifier maps
hand-card crops to eight supported card classes. An ONNX-exported YOLO-style
detector proposes field objects and boxes. The policy is a small MLP that scores
candidate actions from a normalized state plus action features. Rules still
filter candidates for legal elixir, lane, spell, and arena behavior.”

### Why PyTorch?

“PyTorch gave me a straightforward training and inference path, easy checkpoint
loading, and a practical Apple MPS option. I kept policy inference on CPU for
predictability while validating that CPU and MPS outputs agree closely on the
same deterministic input.”

### Why YOLO/OpenCV?

“OpenCV is a good fit for inexpensive fixed regions, color checks, templates,
and screenshot preprocessing. YOLO-style detection is useful when object
locations vary across the arena. Exporting to ONNX makes the runtime dependency
smaller and lets me use a predictable CPU provider.”

### How does the policy decide?

“The state has timing, elixir, side, available-card mask, activity, win
condition, lane, tower-health deltas, and crown features. For every affordable
card, the code generates legal-ish placement candidates. Each candidate gets a
four-value action encoding. The policy scores the resulting 30-value rows, then
the hand-written masks and clamping rules decide what is actually eligible.”

### How did you test it?

“I use unit fixtures for Android dumpsys formats, dry-run input, reward-screen
state transitions, RGB/BGR preprocessing, detector box bounds, policy metric
queue publication, and worker-finish graph retention. I also ran real ADB
screenshot and model-loading checks on BlueStacks Air. I distinguish those
smoke checks from a statistically meaningful live battle-accuracy evaluation.”

### How did you make automation safe?

“All normal ADB click and swipe paths pass through controller methods. In
dry-run, those methods log the exact action and return before issuing shell
input. The same guard exists for the platform-specific MEmu override. I test
that command lists stay empty in dry-run and contain the expected commands in
normal mode.”

### What broke when you revisited it?

“The old project assumed a BlueStacks metadata file, a particular Android
foreground field, old game pixels, and an old reward flow. The current emulator
also had a renderer/display contract that needed to be explicit. Those were
compatibility failures, not reasons to rewrite the whole bot.”

### What would you improve next?

“I would build a captured-frame regression corpus for menu, battle, loading, and
reward states; add detector/classifier precision and recall reports; persist
training metrics; and separate policy training from a tracked release
checkpoint. I would also replace some fixed visual signatures with versioned
templates or a more explicit screen-state model.”

### What would you do differently today?

“I would define typed interfaces between screenshot, perception, state, policy,
and control from the beginning. I would make coordinate calibration and model
versions explicit, use structured logs, and keep live input behind a separate
capability rather than an environment switch alone. The existing project was a
useful experiment, but those boundaries would reduce maintenance cost.”

### What did you learn?

“I learned that reliable automation is mostly about observability, state
transitions, and safe failure handling. A model can be correct while the crop,
color order, activity parser, or emulator surface is wrong. I also learned to
prefer small, testable compatibility fixes over speculative rewrites.”

## Files to review before an interview

| File | Review | Likely question | Do not memorize |
| --- | --- | --- | --- |
| `towerlogic/emulators/bluestacks.py` | Instance/config discovery, private ADB server, renderer, startup readiness, foreground parsing | How do you prove the right device and app are ready? | Every BlueStacks config key |
| `towerlogic/emulators/adb_base.py` | Shared screenshot, click/swipe, app launch, dry-run boundary | Where can input be suppressed or bypassed? | ADB shell syntax by heart |
| `towerlogic/emulators/base.py` | Cross-controller interface and dry-run flag | What does the abstraction guarantee? | Legacy unused methods |
| `towerlogic/bot/nav.py` | Main-menu/battle/reward detectors and recovery state results | How do you avoid relaunch loops? | Every pixel coordinate |
| `towerlogic/bot/fight.py` | Battle loop, perception-to-action handoff, candidate masking | What happens when detection is uncertain? | The full 3,000-line file |
| `towerlogic/bot/card_detection.py` | Crops, templates, class/group maps, classifier fallback | How are card slots mapped to actions? | The large color-data table |
| `towerlogic/detection/hand_classifier.py` | Architecture, transform, checkpoint/class map, device choice | How did you validate preprocessing? | Layer parameter counts |
| `towerlogic/detection/onnx_detector.py` | Preprocess, decode, NMS, class/side handling | Why convert `xywh` before NMS? | ONNX session boilerplate |
| `towerlogic/bot/policy.py` | State/action dimensions, candidate scoring, training update | What does one policy row represent? | Every reward-shaping constant |
| `towerlogic/bot/worker.py` | Process lifecycle, platform registry, config handoff | Why use a worker process? | UI field enum names |
| `towerlogic/utils/logger.py` | Stats snapshots, queue publication, reward telemetry | What does the graph actually show? | Individual counter names |
| `towerlogic/interface/ui.py` | GUI settings and graph rendering | How does worker data reach the canvas? | Tk geometry details |

## Roles and transferable skills

### Game development

TowerLogic is strongest here for state machines, gameplay-system reasoning,
debugging timing-sensitive transitions, and testing recovery paths. The fight
loop, candidate masks, and reward-screen recovery provide concrete examples.
It is weaker as an example of graphics rendering, multiplayer networking, or a
shipping game engine; use another project for those topics.

### Web/software development

Use the package layout, platform adapters, environment-driven paths, optional
dependencies, multiprocessing boundary, regression tests, and README work to
discuss maintainability and compatibility. It is not a web application, so use
another project for HTTP APIs, databases, frontend frameworks, or deployment.

### Game UI/UX

The project demonstrates observing UI state, recognizing when a visual
assumption became stale, distinguishing loading/reward/menu/battle screens, and
recovering from an obstruction. It does not demonstrate user research or
polished product UX; the GUI is an engineering control surface.

### Operations/strategy

The strongest connection is turning ambiguous runtime failures into repeatable
checks: process, foreground, rendered frame, screenshot, detector, and policy
state. Dry-run logging and bounded retries show operational safety. Another
project may be better for business process ownership or team operations.

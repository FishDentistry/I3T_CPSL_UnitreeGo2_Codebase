# dog_voice_commands

Offline voice commands for the Unitree Go2 that respond only to their owner's voice.

Say **"Hey Yeller, sit"** and the Go2 sits. Speech recognition and speaker
verification both run locally (no cloud services), and the package plugs
into the rest of this repository through one ROS 2 topic.

| Say | Go2 action | Sport API |
|---|---|---|
| "Hey Yeller, sit" | Sit | `Sit` (1009) |
| "Hey Yeller, shake" | Raise a front paw to shake | `Hello` (1016) |
| "Hey Yeller, lay down" | Lie down | `StandDown` (1005) |
| "Hey Yeller, stand up" | Return to normal standing, **only if currently sitting or lying down** | From sitting: `RiseSit` (1010). From lying: `StandUp` (1004), then `BalanceStand` (1002) |

The wake phrase and command can be said together or separately: "Hey Yeller"
… *(pause)* … "sit" works too, as long as the command comes within
`arm_timeout_sec` (4 s by default).

### Setup at a glance

Setup is done once per robot. Afterwards, voice commands only need one
argument on the main launch file:

```bash
ros2 launch go2_launcher dog.launch.py launch_voice_commands:=true
```

| Step | When |
|---|---|
| Install dependencies, download the speech model, build | Once |
| Enroll your voice (also saves your microphone choice) | Once; repeat only for a new owner, microphone, or mounting position |
| Choose and save a verification threshold | Once |
| `launch_voice_commands:=true` | Every run |

Everything the setup produces (speech model, voice profile, and settings)
is stored under `~/.local/share/dog_voice_commands/`, outside the
repository. It survives rebuilds and `git pull`s.

---

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Enrolling your voice](#enrolling-your-voice)
- [Choosing a verification threshold](#choosing-a-verification-threshold)
- [Running](#running)
- [Robot integration](#robot-integration)
- [Configuration](#configuration)
- [ROS interface](#ros-interface)
- [Customizing commands](#customizing-commands)
- [Troubleshooting](#troubleshooting)
- [Limitations and safety](#limitations-and-safety)
- [Development](#development)
- [License and acknowledgments](#license-and-acknowledgments)

---

## How it works

```
 microphone ──► Vosk recognizer ──► wake/command ──► speaker ──► /go2/sport_cmd ──► robot
  (16 kHz)     (fixed grammar)       parser         verifier     (Go2Cmd)          connection
                                                       ▲
                                     enrolled voice ───┘
                                     profile (.npz)
```

1. **Speech recognition.** [Vosk](https://alphacephei.com/vosk/) decodes
   microphone audio against a small fixed grammar: the wake phrases, the
   command phrases, and `[unk]` for everything else. Unrelated speech is
   rejected as `[unk]` instead of being forced onto the closest command,
   which keeps false triggers low and CPU use small enough for the Go2
   expansion board.
2. **Command parsing.** Transcripts are matched against
   [`config/commands.yaml`](config/commands.yaml). A command is accepted
   only after a wake phrase.
3. **Speaker verification.** The audio of the matched utterance is converted
   into a speaker embedding (ECAPA-TDNN from
   [SpeechBrain](https://speechbrain.github.io/) by default) and compared
   against your enrolled voice profile using cosine similarity. Commands from
   other voices are rejected.
4. **Output.** Accepted commands are published as
   `go2_interfaces/msg/Go2Cmd` messages containing the Go2 sport-mode API
   ID. This node never connects to the robot itself; see
   [Robot integration](#robot-integration).

Posture-dependent commands, such as "stand up", also check the robot's
current posture, which the Go2 publishes on `lf/sportmodestate`, before
running. They are refused if the robot is in any other posture, or if no
recent posture data is available.

A cooldown (3 s by default) stops one command from firing repeatedly.

---

## Requirements

**Hardware**

- A Unitree Go2 (tested setup: Go2 EDU) set up as described in the
  [repository README](../../README.md).
- A USB microphone plugged into the Go2 expansion board (the reference
  setup, in which the node runs on the expansion board). A directional or
  noise-cancelling microphone, mounted away from the fans, helps a lot,
  since the Go2's fans and motors are loud. Any computer on the same ROS 2
  network as the robot can run the node instead, using its own microphone.

**Software**

- Ubuntu 20.04 with ROS 2 Foxy (other ROS 2 distributions should work but
  are untested).
- Python 3.8 or newer.
- The `go2_interfaces`, `unitree_go`, and `dog_utilities` packages from
  this repository (`dog_utilities` provides the `go2WebRtcBridge` node that
  delivers commands to the robot).
- The Go2's own ROS 2 topics visible on the expansion board, as set up in
  the [repository README](../../README.md#go2-expansion-board-networking-setup).
  "Stand up" reads `lf/sportmodestate`; check with
  `ros2 topic echo /lf/sportmodestate --once`.
- PortAudio, Vosk, sounddevice, and a speaker-embedding backend (installed
  below).

---

## Installation

All commands assume the repository root is
`~/I3T_CPSL_UnitreeGo2_Codebase`; adjust paths to match your checkout.

### 1. System packages

```bash
sudo apt update
sudo apt install -y libportaudio2 python3-pip python3-numpy python3-scipy python3-yaml
```

### 2. Python packages

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase/src/dog_voice_commands
python3 -m pip install --user vosk sounddevice
```

Then install **one** speaker-verification backend.

**SpeechBrain (recommended, more accurate).** Requires PyTorch and
torchaudio.

```bash
python3 -m pip install --user "speechbrain>=1.0.0,<1.1"
```

SpeechBrain 1.1 and newer require PyTorch 2.7.1+, which is not available for
Ubuntu 20.04 on Jetson, so stay on the 1.0 series there. On x86 machines,
`pip` installs a compatible PyTorch automatically.

**Resemblyzer (lighter alternative).** Requires PyTorch only, so it is easier
on boards where torchaudio is hard to install. It is less discriminative and
needs a higher threshold.

```bash
python3 -m pip install --user resemblyzer
```

> **Go2 expansion board (Jetson Orin) note:** do not let `pip` install
> PyTorch from PyPI on the Jetson; that build has no GPU support and may not
> match JetPack. Install NVIDIA's PyTorch wheel for your JetPack version
> first (see NVIDIA's *Installing PyTorch for Jetson Platform* guide), then
> install the backend. The CPU is fast enough for verification, so
> `compute_device: cpu` (the default) is fine.

[`requirements.txt`](requirements.txt) lists the same dependencies for
reference.

### 3. Download the speech model

The node expects the small English Vosk model (about 40 MB). Small models
support the runtime grammars this package relies on; the large models do
not.

```bash
mkdir -p ~/.local/share/dog_voice_commands/models
cd ~/.local/share/dog_voice_commands/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip
```

To keep the model somewhere else, set `vosk_model_path`.

### 4. Build

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-up-to dog_voice_commands
source install/setup.bash
```

### 5. Find your microphone

```bash
ros2 run dog_voice_commands list_audio_devices
```

Run this on the expansion board with the USB microphone plugged in. Note a
distinctive part of your microphone's name (for example `USB`). You will pass
it to `enroll_speaker` as `--device`, which then saves it for every later
launch. Prefer part of the name over the index, because indices can change
when devices are plugged in or the board reboots. If no device is given, the
system default input is used.

---

## Enrolling your voice

Speaker verification needs a profile of the owner's voice. Enrollment records
about a dozen short phrases and stores their speaker embeddings.

```bash
ros2 run dog_voice_commands enroll_speaker --device USB
```

You will be prompted to say phrases such as "Hey Yeller, sit". For each one,
press **Enter** and speak within the 3-second recording window.

For the best results:

- **Use the microphone and position you will actually use.** Enroll on the
  expansion board with the microphone mounted on the robot, the robot
  powered on (so fan noise is in the recordings), and you standing where
  you normally give commands from, typically 1-2 m away.
- **Speak naturally** at your normal commanding volume, and vary your
  tone a little between takes.
- **Only the owner should speak** during enrollment.
- **More samples help.** Use `--samples 20` for a more robust profile, or
  add samples later with `--append` (for example, from a different room).

When it finishes, the tool prints how consistently your samples match each
other. It also saves the microphone (`--device`, `--capture-rate`) and the
backend to the per-machine settings file (see
[Configuration](#configuration)), so you do not need to pass them when
launching. The first run downloads the embedding model (about 80 MB for
SpeechBrain) to `~/.local/share/dog_voice_commands/models/`.

**Useful options**

| Option | Purpose |
|---|---|
| `--device NAME` | Microphone to record with; also saved as `audio_device` for launches. |
| `--samples N` | Number of recordings (default 12). |
| `--seconds S` | Length of each recording (default 3.0). |
| `--backend resemblyzer` | Enroll for the Resemblyzer backend. It must match `speaker_backend` at run time. |
| `--append` | Add samples to an existing profile. |
| `--from-dir DIR` | Enroll from existing 16-bit PCM WAV files instead of recording, for example recordings made on another computer. |
| `--save-audio DIR` | Keep the trimmed recordings as WAV files. |
| `--capture-rate 48000` | For microphones that cannot record at 16 kHz. |
| `--profile PATH` | Profile location (default `~/.local/share/dog_voice_commands/speaker_profile.npz`). |

> **Privacy.** The profile contains voice embeddings, a biometric
> representation of your voice. It is stored outside the repository with
> owner-only file permissions. Do not commit it, and do not commit any
> recordings saved with `--save-audio`.

---

## Choosing a verification threshold

A command is accepted when its similarity score is at or above
`speaker_threshold`. Defaults are **0.45** for SpeechBrain and **0.75** for
Resemblyzer. These are reasonable starting points, but the right value
depends on your microphone, room, and voice, so check it:

```bash
ros2 run dog_voice_commands verify_speaker --device USB
```

Record yourself several times, then have one or two other people do the
same. Each take prints its score and an ACCEPT/REJECT verdict. Choose a
threshold that sits clearly below your own scores and clearly above
everyone else's. If the two ranges overlap, enroll more samples in the
real environment before raising the threshold.

- **Raise the threshold** if other people are accepted.
- **Lower it** if you are rejected too often.

Save your choice so every launch uses it:

```bash
ros2 run dog_voice_commands verify_speaker --save-threshold 0.5
```

To try a candidate value before saving it, pass `--threshold 0.5` to a normal
`verify_speaker` run. `verify_speaker` also accepts WAV files:
`verify_speaker take1.wav take2.wav`.

---

## Running

### 1. Dry run (no robot motion)

Start with a dry run to check recognition and verification without sending
anything to the robot:

```bash
ros2 launch dog_voice_commands voice_commands.launch.py dry_run:=true
```

In a second terminal, watch what the node hears and decides:

```bash
ros2 topic echo /voice_command_node/events
```

Each recognized utterance produces a JSON event, for example:

```json
{"event": "dry_run", "text": "hey yeller sit", "command": "sit", "api_ids": [1009], "confidence": 0.94, "speaker_score": 0.62}
{"event": "wrong_posture", "text": "hey yeller stand up", "command": "stand_up", "posture": "standing", "confidence": 0.93, "speaker_score": 0.58}
{"event": "rejected_speaker", "text": "hey yeller sit", "command": "sit", "confidence": 0.91, "speaker_score": 0.18, "reason": "score 0.180 below threshold 0.450"}
```

### 2. Live, with the main robot launch

The simplest way to run voice commands for real is through the repository's
main launch file, which also starts the `go2WebRtcBridge` node that delivers
commands to the robot:

```bash
ros2 launch go2_launcher dog.launch.py launch_voice_commands:=true
```

The microphone and threshold come from the settings saved during setup. To
use a different microphone for one run without changing the saved setting,
add `voice_audio_device:=<name>`. All of the main launch file's other
arguments (`internal_board_ip`,
`launch_arm`, and so on) work as usual. Make sure the Go2 is standing on
open ground and the remote control is in reach before speaking a command.

### 3. Live, standalone

If `dog.launch.py` is already running without voice commands, start the
voice node on its own; it will use the bridge that is already running:

```bash
ros2 launch dog_voice_commands voice_commands.launch.py
```

The standalone launch accepts `audio_device`, `dry_run`, and
`speaker_verification` arguments for one-off overrides, and
`params_file:=/path/to/params.yaml` to replace the packaged defaults.

### Muting

Voice commands can be paused and resumed at run time without restarting the
node:

```bash
ros2 topic pub --once /voice_command_node/enable std_msgs/msg/Bool "{data: false}"   # mute
ros2 topic pub --once /voice_command_node/enable std_msgs/msg/Bool "{data: true}"    # unmute
```

---

## Robot integration

The Go2 internal board accepts only one WebRTC client at a time. In this
repository that connection is owned by a single node, `go2_webrtc_bridge`
(executable `go2WebRtcBridge` in `dog_utilities`), and every other node
sends commands through it:

```
voice_command_node ──► /go2/sport_cmd   (Go2Cmd) ──┐
                                                    ├──► go2_webrtc_bridge ──► Go2 internal board
cmd_vel_translator ──► /go2/webrtc_send (String) ──┘         (WebRTC)
```

`voice_command_node` publishes the sport-mode API ID of each accepted
command as a `go2_interfaces/msg/Go2Cmd` on `/go2/sport_cmd`. The bridge
forwards only IDs listed in `ROBOT_CMD`
(`dog_utilities/scripts/go2_constants.py`) and logs every command it
sends. The robot IP is set on the bridge through the `internal_board_ip`
argument of `dog.launch.py`.

To run the bridge without the main launch file:

```bash
ros2 run dog_utilities go2WebRtcBridge --ros-args -p internal_board_ip:=192.168.123.161
```

You can send the same commands by hand, which is useful for testing the
robot side without speaking:

```bash
ros2 topic pub --once /go2/sport_cmd go2_interfaces/msg/Go2Cmd "{cmd: 1009}"
```

### Standing back up

"Hey Yeller, stand up" returns the robot to its normal standing state
(`BalanceStand`, the state it is in after booting) from the postures the
other commands leave it in:

| Current posture | Actions sent |
|---|---|
| Sitting (after "sit") | `RiseSit` (1010) |
| Lying (after "lay down") | `StandUp` (1004), then `BalanceStand` (1002) `step_delay_sec` later |
| Anything else | Nothing; the command is refused |

The posture comes from `SportModeState.mode`: 10 (sit) counts as sitting,
and 5 (lieDown) counts as lying. Mode 6 (jointLock), which both
`StandDown` and `StandUp` produce, counts as lying when `body_height` is
below `lying_height_m` (0.2 m by default) and as standing otherwise.
Damping mode (7), the soft emergency stop, is deliberately excluded:
recover from it with the remote.

Shake (`Hello`) is only performed from standing, and returns to standing on
its own. Sport API
behaviour can differ between Go2 firmware versions; check the IDs in
`dog_utilities/scripts/go2_constants.py` against your firmware if an action
does not run.

---

## Configuration

Parameters are loaded from three places, with later ones taking
precedence:

1. **Packaged defaults:** [`config/voice_commands.yaml`](config/voice_commands.yaml).
   Avoid editing this for a single robot, since it is part of the repository.
2. **Per-machine settings:** `~/.local/share/dog_voice_commands/params.yaml`.
   `enroll_speaker` and `verify_speaker --save-threshold` write this file
   for you. You can also edit it by hand to change any parameter below, for
   example:

   ```yaml
   voice_command_node:
     ros__parameters:
       audio_device: USB
       speaker_threshold: 0.5
       cooldown_sec: 2.0
   ```
3. **Launch arguments** that are set explicitly (`audio_device` /
   `voice_audio_device`, `dry_run`, `speaker_verification`).

| Parameter | Default | Description |
|---|---|---|
| `commands_file` | *(packaged `commands.yaml`)* | Vocabulary file with wake phrases and commands. |
| `vosk_model_path` | `~/.local/share/dog_voice_commands/models/vosk-model-small-en-us-0.15` | Vosk model directory. Must support runtime grammars. |
| `min_word_confidence` | `0.6` | Utterances with a lower mean word confidence are ignored. |
| `audio_device` | `""` | Microphone index or name substring; empty means system default. |
| `capture_sample_rate` | `16000` | Rate to open the microphone at. Audio is resampled to 16 kHz. |
| `arm_timeout_sec` | `4.0` | Time allowed between a lone wake phrase and the command. |
| `cooldown_sec` | `3.0` | Minimum time between accepted commands. |
| `speaker_verification` | `true` | Require the enrolled owner's voice. If `false`, **anyone** can command the robot. |
| `speaker_backend` | `speechbrain` | `speechbrain` or `resemblyzer`. Must match the profile. |
| `speaker_profile_path` | `~/.local/share/dog_voice_commands/speaker_profile.npz` | Enrolled voice profile. |
| `speaker_threshold` | `-1.0` | Acceptance threshold; a negative value uses the backend default (0.45 / 0.75). |
| `compute_device` | `cpu` | Torch device for the embedding model (`cpu` or `cuda`). |
| `sport_state_topic` | `lf/sportmodestate` | Robot state used by posture-dependent commands. |
| `state_timeout_sec` | `2.0` | Posture data older than this counts as unknown, and posture-dependent commands are refused. |
| `lying_height_m` | `0.2` | Body height below which a joint-locked robot counts as lying. |
| `step_delay_sec` | `2.5` | Time between actions in a multi-step command (for example `StandUp` → `BalanceStand`). |
| `command_topic` | `/go2/sport_cmd` | Output topic for `Go2Cmd` messages. |
| `start_enabled` | `true` | Whether the node accepts commands at startup. |
| `dry_run` | `false` | Recognize and log commands without publishing them. |

If speaker verification is enabled and no profile exists, the node exits at
startup with instructions rather than silently accepting every voice.

---

## ROS interface

**Published**

| Topic | Type | Description |
|---|---|---|
| `/go2/sport_cmd` | `go2_interfaces/msg/Go2Cmd` | Sport-mode API ID of each accepted command. |
| `~/events` | `std_msgs/msg/String` | JSON diagnostic event for each recognized utterance. |

Event types: `accepted`, `dry_run`, `armed` (wake phrase heard, waiting for
command), `rejected_speaker`, `cooldown`, `expired` (command came after the
arm window), `low_confidence`, `wrong_posture` (a posture-dependent command
was heard in a posture it does not run from), and `no_robot_state` (no recent
`SportModeState` was available).

**Subscribed**

| Topic | Type | Description |
|---|---|---|
| `~/enable` | `std_msgs/msg/Bool` | Mute (`false`) or unmute (`true`) voice commands. Muting also cancels any queued steps of a multi-step command. |
| `lf/sportmodestate` | `unitree_go/msg/SportModeState` | Robot posture, published by the Go2 (topic set by `sport_state_topic`). |

**Executables**

| Executable | Description |
|---|---|
| `voice_command_node` | The voice command node. |
| `enroll_speaker` | Record samples and build the owner's voice profile. |
| `verify_speaker` | Score live speech or WAV files against the profile. |
| `list_audio_devices` | List available microphones. |

---

## Customizing commands

The vocabulary is defined in [`config/commands.yaml`](config/commands.yaml):

```yaml
wake_phrases:
  - hey yeller
  - hey yellow      # common mis-hearing of "yeller"

commands:
  sit:
    api_id: 1009
    phrases: [sit, sit down]
```

- Every word must exist in the Vosk model's dictionary. At startup the node
  logs and disables any phrase that uses an unknown word. "Yeller" is not a
  standard word, which is why the default file also accepts phrases such as
  "hey yellow" that the model may hear instead. Remove an alias if it
  causes false triggers.
- A phrase may belong to only one command.
- Each command defines exactly one of:
  - `api_id`: a Go2 sport-mode API ID, sent whenever the command is heard.
  - `from_posture`: the postures the command may run from (`standing`,
    `sitting`, `lying`, `damped`, `other`), each with the API IDs to send in
    order. The command is refused in any other posture. For example:

    ```yaml
    stand_up:
      phrases: [stand up]
      from_posture:
        sitting: [1010]       # RiseSit
        lying: [1004, 1002]   # StandUp, then BalanceStand
    ```
- API IDs are listed in `dog_utilities/scripts/go2_constants.py`
  (`ROBOT_CMD`); the bridge refuses any ID not in that list. Only add
  actions that are safe to trigger by voice in your environment.

To use your own file without editing the package, copy it and set
`commands_file`. After changing the vocabulary you do not need to re-enroll;
the voice profile is independent of the words used.

---

## Troubleshooting

**`Vosk model not found`**
Download the model ([Installation, step 3](#3-download-the-speech-model)) or
point `vosk_model_path` at it.

**`No speaker profile at ...`**
Run `enroll_speaker` first. To test recognition without verification, launch
with `speaker_verification:=false` and `dry_run:=true`.

**The USB microphone is not listed**
Unplug and replug it, then check `arecord -l`. If it appears there but not
in `list_audio_devices`, make sure the user is in the `audio` group
(`sudo usermod -aG audio $USER`, then log out and back in).

**`Invalid sample rate` or `Error opening InputStream`**
The microphone does not support 16 kHz capture. Set
`capture_sample_rate: 48000` (or `44100`), and pass the same value to
`enroll_speaker` / `verify_speaker` with `--capture-rate`.

**Nothing is recognized**
Check the input with `arecord -l` and `arecord -D plughw:<card>,0 -f S16_LE -r 16000 -d 3 test.wav`
followed by `aplay test.wav`. Make sure the right device is selected and its
gain is not muted (`alsamixer`). Watch `~/events` for `low_confidence`
events; if they appear, lower `min_word_confidence` slightly or move the
microphone closer.

**The wake phrase is not recognized but commands are**
The model may be hearing "Yeller" as a different word. Try saying it more
distinctly, or add an alias for what is being heard to `wake_phrases`.

**My commands are rejected as the wrong speaker**
Run `verify_speaker` to see your scores. Re-enroll (or `--append` more
samples) using the same microphone, position, and background noise as real
use, or lower `speaker_threshold`.

**"Stand up" is always refused with `no_robot_state`**
The node is not receiving `SportModeState`. Check
`ros2 topic echo /lf/sportmodestate --once` on the machine running the
node. If nothing arrives, revisit the expansion-board network and
CycloneDDS setup in the [repository README](../../README.md).

**"Stand up" is refused with `wrong_posture` while the robot is lying down**
Check the posture the node sees in the event, and compare it with
`ros2 topic echo /lf/sportmodestate --field mode --once` and
`--field body_height`. If the robot is joint-locked (mode 6) and its body
height is above `lying_height_m`, raise `lying_height_m` in the
per-machine settings file.

**"Stand up" from lying stops with the robot standing stiffly**
`StandUp` locks the joints; `BalanceStand` follows `step_delay_sec` later.
If the robot has not finished rising when `BalanceStand` arrives, increase
`step_delay_sec`.

**Commands work in the dry run but the robot does not move**
Confirm that messages arrive with `ros2 topic echo /go2/sport_cmd`, and
check the `go2_webrtc_bridge` output: it logs `Sent sport command ...` for
each command, or explains why a command was dropped (for example, if the
connection to the robot failed at startup). Make sure only one bridge is
running (see [Robot integration](#robot-integration)). Some actions only run from
certain postures; for example, `Hello` requires the robot to be standing.

---

## Limitations and safety

- **Speaker verification is a convenience, not a security control.** It
  reliably ignores bystanders and stops most accidental triggers, but it can
  be fooled by a recording of the owner's voice or by a sufficiently similar
  voice, and it can occasionally reject the owner. Do not rely on it to
  prevent unauthorized control of the robot.
- **Keep the remote control within reach** and the robot in a clear area
  whenever voice commands are enabled.
- Recognition degrades with distance, wind, and loud background noise,
  including the Go2's own fans. A microphone close to the speaker works
  best.
- Only English is supported out of the box. Other Vosk small models can be
  used by changing `vosk_model_path` and the phrases in `commands.yaml`.

---

## Development

Unit tests cover phrase parsing, posture classification and
posture-dependent commands, audio buffering and resampling, and the
speaker-profile logic. They do not need a microphone, ROS graph, or models:

```bash
cd ~/I3T_CPSL_UnitreeGo2_Codebase
colcon test --packages-select dog_voice_commands
colcon test-result --verbose
```

or directly with `python3 -m pytest src/dog_voice_commands/test`.

Package layout:

```
dog_voice_commands/
├── config/
│   ├── commands.yaml          # Wake phrases and commands
│   └── voice_commands.yaml    # Node parameters
├── dog_voice_commands/
│   ├── audio.py               # Microphone capture, ring buffer, WAV helpers
│   ├── command_parser.py      # Wake/command matching (no ROS dependency)
│   ├── posture.py             # SportModeState -> posture classification
│   ├── recognizer.py          # Vosk grammar recognizer
│   ├── speaker_verifier.py    # Embedding backends and voice profiles
│   ├── tools.py               # enroll_speaker, verify_speaker, list_audio_devices
│   └── voice_command_node.py  # ROS 2 node
├── launch/voice_commands.launch.py
└── test/
```

---

## License and acknowledgments

This package is released under the Apache License 2.0; see
[`LICENSE`](LICENSE).

It builds on:

- [Vosk](https://github.com/alphacep/vosk-api) and its English models
  (Apache 2.0), for offline speech recognition.
- [SpeechBrain](https://github.com/speechbrain/speechbrain) (Apache 2.0) and
  its pretrained [ECAPA-TDNN speaker model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb),
  for speaker verification.
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (Apache 2.0), as
  an alternative speaker-embedding backend.

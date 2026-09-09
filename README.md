# MiniMax H3 Sliding Window Anchor

A [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) plugin.

MiniMax H3 Sliding Window Anchor injects an end frame at the beginning of a
new sliding window to create smooth transitions between windows, and can hold
colour, saturation and brightness steady across the join.

## What it does

When Wan2GP generates a long video, it does so in sliding windows — each window
is a separate generation. This plugin takes the last frame of the window just
produced and feeds it back as a condition on the first frame of the next one,
so the new window starts from where the previous one ended.

That fixes the content of the join but not its grade. The model re-synthesises
the anchor frame rather than copying it, and the small error in that
reconstruction is inherited by every window after it, so a long generation
slowly washes out. [Colour consistency](#colour-consistency) measures that
drift and corrects it.

Anchoring runs automatically once enabled. Colour matching is off by default
and has its own section in the panel.

Everything the plugin carries — the anchor frame, the window count, the colour
reference — belongs to one video. Wan2GP runs a whole queue through the same
process back to back, so the plugin reads the window number out of each call
and clears that state whenever the next item in the queue begins, or the next
repeat of the same item. Windows of one video continue from each other;
separate videos do not.

## Which models

MiniMax H3 only.

The position the plugin hands to the model is not a plain frame number. H3
subtracts the number of carried-over overlap frames from it before using it,
so the plugin adds that count back on to land the anchor on the window's first
frame. Other model families read the position literally, which would place the
anchor part-way into the window instead of at its start, and would do so
silently. Rather than guess, the plugin only attaches itself to H3 and leaves
every other model untouched.

## Install

1. Copy the `wan2gp-sliding-window-anchor` folder into `Wan2GP/plugins/`.
2. Enable it in the Plugins tab and save.
3. Restart Wan2GP.

On restart, check the console for:

```
[SlidingWindowAnchor] hooked MiniMax H3
```

If that line is missing the plugin is not active, and nothing else in the
interface will make a difference.

The panel appears as a collapsed accordion named **H3 Sliding Window Anchor**, just above **Generate**. It is shown for every model, but only acts on H3.

## Settings to leave alone

Keep **Discard Last Frames of a Window** and **Trim First Frames** at 0.

Both trim frames after a window has been generated. The anchor is taken from
the end of the raw window, so trimming would leave it pointing at a frame that
is no longer where the next window continues from, and the join it is meant to
smooth would be the one place it no longer matches.

## The panel

- **Enable Sliding Window Anchor** — on/off.
- **Save Injected Frames** — keeps a copy of every frame the plugin injects,
  in the `saved_frames` folder inside the plugin folder, grouped by run.
  Off by default; frames are otherwise discarded when a generation finishes.
- **Anchor Frame** — the frame that will be fed into the next window.
- **Sliding Window** — which window that frame came from.
- **Frame (approx)** — roughly where that frame sits in the finished video.
  Approximate because the final frame numbering depends on trimming Wan2GP
  applies after each window is generated.

Both read `N/A` until the first window has finished, since there is no anchor
frame before then.

## Colour consistency

The model does not copy the anchor frame into the new window, it
re-synthesises it, and that reconstruction is never exact. Blacks lift,
contrast softens, colour desaturates a little. Because every window is
conditioned on the end of the one before it, that error is inherited and
compounds, so a long generation slowly washes out.

**Match Colour Across Windows** measures the difference across each join and
corrects it. It is off by default: whole-window scope rewrites the frames
Wan2GP saves, and a plugin update should not quietly start altering the
output of a setup that was already working.

The drift is measured between the last few frames of the previous window and
the first few *generated* frames of the new one. Those frames cover the same
moment, so what differs between them is reconstruction error rather than the
scene moving on. Comparing whole windows would read a camera panning from a
bright room to a dark one as drift and flatten it out.

### The controls

- **Correction Scope** — **Whole window** corrects the frames Wan2GP goes on
  to save, so the join in the finished video matches. **Anchor frame only**
  leaves the video untouched and corrects just the frame fed into the next
  window; that stops the drift compounding but does not remove what is
  already in the output.
- **Match Against** — **Previous window** matches each window to the one
  before it. **First window** matches every window to the opening one.
- **Brightness / Contrast / Saturation / Colour Cast** — the four axes the
  correction splits into, each switchable on its own.
- **Correction Strength** — how much of the measured difference to remove.
  Below 1 leaves some drift in place, which can look more natural than a
  hard match.
- **Scene Change Threshold** — past this, a difference is read as a cut
  rather than drift and the window is left alone. Real drift is around a
  percent per window, so a difference several times larger is not severe
  drift; it is evidence the frames either side of the join are not showing
  the same moment. Such a measurement is discarded rather than capped.
  Capping it and applying it anyway drags a new scene bodily toward the
  grade of the old one, which is worse than doing nothing.
- **Last correction** — what the most recent correction did, in the same
  terms the console reports.

### Which reference to match against

**Previous window** is the default and is usually the right one. Locking
every window to the first sounds like it would hold a look more tightly, but
on material whose lighting genuinely changes it fights the content and ends
up drifting further than matching the previous window does. Use **First
window** only where the lighting really is meant to be constant throughout.

### How it works

The correction is a per-channel affine in Rec. 709 Y'CbCr, which splits into
exactly the four axes above: the luma mean is brightness, its spread is
contrast, the chroma means are colour cast and their spread is saturation.
Saturation uses one scale for both chroma channels rather than one each,
because scaling them separately would shift hue while claiming to change only
saturation.

Y'CbCr is a linear transform of RGB, so the whole correction collapses into a
single 3x3 matrix and offset applied straight to RGB. CIELAB would track
perceived brightness more faithfully, but converting a whole window through it
costs a cube root per value and several full-size intermediates; on the size
of drift being corrected here, a few percent, that is not worth it.

### Continuing a video

A Continue Video run hands its first window a strip of frames from the end of
the video being continued. Its last frame becomes the first anchor, so the
new window starts from where the old video ended. Every other anchor is the
last frame of a window this plugin watched being generated; a continuation's
first window has no such predecessor, so without this it was the one window
that began with nothing to anchor to -- across the join between old video and
new, which is the most visible join there is.

Those carried-in frames are also the colour reference for that window, and
the reference there is widened. Both sides of a continuation join sit in the
same window, so more frames can be measured on each -- which is not possible
at an ordinary window join, where the previous window's frames are gone by
the time the next one arrives and only the statistics survive.

The span grows symmetrically and stops at whichever comes first: a cut in the
old video, a cut in the new window, the end of the carried-in frames, the
length of the window, or the fixed span used at every other join. Symmetry is the
point -- a swathe of old footage measured against a sliver of new would read
every camera move between them as drift. The console reports what was used
and which limit bound it:

```
continuing from an existing video: 120 frames carried in, measuring 48
either side of the join (limited by the 48-frame maximum)
```

That line also answers how many frames Wan2GP actually hands in, which
decides whether the maximum is doing anything at all. Those frames are a real reference, and the join
between old and new video is the most visible one there is, so the plugin
matches the first window to them. Without this it would be the one join never
checked.

That reference is a short strip rather than a whole window, so by default
only the anchor is corrected there and the video itself is left as generated.
**Anchor Only On A Continued Video's First Window** turns that off if you
want the first window corrected too. Later windows in the same run behave
normally either way.

A run that starts from scratch has no carried-over frames, so its first
window still defines the look and is never corrected.

### Why there is no span control

How many frames either side of a join get compared is fixed at five. It was a
setting; the measurements took it away. On real footage a join with a visible
change reads just as strongly at five frames as at eighteen, while a join
where nothing happens reads more false difference the wider the span gets.
Worse, a large span used to widen the noise floor's own sampling stretch,
which is exactly what made it blind to real steps. There is no value worth
offering, so there is no control.

### Finding cuts

Fast motion produces large frame-to-frame changes as well, so a cut is
recognised by its neighbours being ordinary rather than by its size alone.
On measured footage containing four hand-checked cuts, the cuts scored 9.3 to
21.3 against their neighbours while the largest change caused by motion alone
scored 3.6; the threshold sits at 6.

### Running it without anchoring

Colour matching and frame anchoring are independent. Matching measures the
difference across a join and corrects the window; it needs a join, not an
injected frame. Switching **Enabled** off leaves colour matching running on
its own, which is the cleanest way to see which of the two is doing what on
a given piece.

The one combination that cannot work is **Anchor frame only** scope with
anchoring switched off -- the correction would be applied to a frame that is
never written and never injected. That falls back to whole-window scope, and
says so once.

### The noise floor

Real footage moves. Grain, a passing highlight and the scene itself all shift
these statistics from frame to frame, and on dark or low-chroma material that
wobble is often several times larger than the drift being looked for.

So before correcting anything, the plugin measures how much the same
statistics vary *inside* the current window, where there is no join and the
true answer is therefore no change. Anything at the join not clearly larger
than that wobble is held back. A difference the size of the noise is
discarded entirely; one far larger passes through almost untouched.

This matters more than it sounds. The correction is applied every window, and
each window inherits the last, so chasing noise does not average out over a
long generation -- it accumulates into exactly the drift the feature exists
to prevent. On measured 1376x576 footage at the default span, a raw
measurement of x0.943 saturation was entirely noise, and shrinkage correctly
reduced it to no correction at all.

The console prints the noise floor next to each correction. If the correction
is not comfortably larger than the floor beside it, nothing is being measured
except wobble.

### When it is not helping

- **The console reports clipping.** A correction strong enough to push values
  past the ends of the range is flattening highlights or shadows rather than
  shifting them. Lower **Correction Strength** or **Maximum Correction**.
- **A window contains a cut.** Handled two ways. A cut at the join makes the
  measurement meaningless, so the window is skipped and the console says so;
  the next window measures the new scene against itself and normal work
  resumes. A cut part way through a window is found and the correction stops
  there, since the new scene has nothing to be matched to yet. The seam that
  leaves falls on the cut, where it cannot be seen.
- **Every window is rejected as a scene change.** On material with a lot of
  motion the chroma statistics swing several percent between neighbouring
  groups of frames on their own, which trips the threshold at every join.
  Raising it lets corrections through, but also lets real scene changes
  through, and on such material the drift is smaller than the variation
  anyway. Leaving colour matching off is the better answer.
- **The correction is always nothing, and the noise floor is large.** The
  drift on that material is smaller than the measurement can resolve. That
  is a real answer, not a failure: there is nothing there worth correcting.
- **Drift develops inside a single window.** The correction is one constant
  per window, so it pulls the window as a whole into line but will not
  flatten it internally. There is nothing to measure the far end of a window
  against.

## Notes

- Anchor frames are written to `state/` inside the plugin folder and cleared
  when a video finishes, when the queue finishes, when Wan2GP shuts down, and
  on startup. Nothing carries over between videos unless **Save Injected
  Frames** is on, which keeps copies in `saved_frames/`, one folder per video.
- The boundary between two videos is found from the generate call itself, not
  from the panel. Wan2GP numbers the windows of every video from 1, so window
  1 marks a new video however the queue reached it. The panel polls on a
  timer, is not there at all if it failed to place, and only ever sees the
  queue as a whole stop — which is why it used to miss every boundary inside
  one. A build that passes no window number falls back to the frame the
  window begins at, and then to whether any frames were carried in; the last
  of those cannot recognise a queued **Continue Video** run, and the console
  says which signal was used.
- The plugin patches Wan2GP's model pipelines at runtime. If a future version
  moves or renames one, the plugin says so in the console and steps aside;
  generation continues unaffected.
- With colour matching off, or set to **Anchor frame only**, the plugin reads
  the frames a window produces and adds one image to what the model is given;
  it does not alter the video that Wan2GP saves. **Whole window** scope is the
  one setting that does, by design — it is what makes the join match in the
  finished video rather than only in the frame handed to the next window.
- A window is corrected in place, in chunks sized by value count rather than
  frame count, so the working set stays around 125 MB whether the output is
  480p or 4K. Correction runs after `generate()` has returned, so denoising
  and VAE decode are both finished and their working memory already freed.
- H3 generates under `torch.inference_mode`, and PyTorch refuses in-place
  writes to the tensors that come out of it -- but only from outside
  inference mode. The plugin re-enters it, which needs no second copy of the
  window, and means the correction reaches the saved video whether or not
  the pipeline re-reads its result dictionary. A tensor that refuses writes
  for some other reason falls back to a corrected copy; which path is taken
  is settled by a probe write before any real work, so a window can never
  end up corrected at one end and not the other.

## Requirements

`numpy`, `Pillow` and `torch`, all already installed with Wan2GP.

## Tests

```
cd wan2gp-sliding-window-anchor
python3 test_anchor.py
```

100 checks, covering the injection arithmetic, the four window formats
Wan2GP hands back, the colour maths, and a simulated eight-window run in
which drift is shown to compound without correction and not to with it.

## Licence

MIT — see [LICENSE](LICENSE).

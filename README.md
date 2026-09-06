# MiniMax H3 Sliding Window Anchor

A [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) plugin.

MiniMax H3 Sliding Window Anchor injects an end frame at the beginning of a
new sliding window to create smooth transitions between windows.

## What it does

When Wan2GP generates a long video, it does so in sliding windows — each window
is a separate generation. This plugin takes the last frame of the window just
produced and feeds it back as a condition on the first frame of the next one,
so the new window starts from where the previous one ended.

It runs automatically once enabled. There is nothing to configure beyond
switching it on.

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

## Notes

- Anchor frames are written to `state/` inside the plugin folder and cleared
  when a generation finishes, when Wan2GP shuts down, and on startup. Nothing
  carries over between runs unless **Save Injected Frames** is on, which keeps
  copies in `saved_frames/`.
- The plugin patches Wan2GP's model pipelines at runtime. If a future version
  moves or renames one, the plugin says so in the console and steps aside;
  generation continues unaffected.
- The plugin only reads the frames a window produces and adds one image to
  what the model is given. It does not alter the video that Wan2GP saves.

## Requirements

`numpy` and `Pillow`, both already installed with Wan2GP.

## Licence

MIT — see [LICENSE](LICENSE).

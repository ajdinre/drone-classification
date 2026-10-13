# Drone Image Classification

Analyze a still image to answer two questions: **Does it contain a drone?** If so, **which drone model is it?**

## Input and output

- **Input:** One JPEG or PNG image.
- **Output:** `contains_drone` (`true` or `false`) and `model` (one of the ten target models, `unknown`, or `null`).
- Use `null` when no drone is visible. Use `unknown` when a drone is visible but its model is outside the target set or cannot be identified reliably from the image.
- The first version focuses on images with one primary drone. If several drones are visible, it should still report that a drone is present, but may return `unknown` for the model.

Example:

```json
{
  "contains_drone": true,
  "model": "DJI Mini 4 Pro"
}
```

## Initial target models

These ten models are a starting set of frequently observed consumer drones, selected from the model counts in a [2025 U.S. drone traffic study](https://assureuas.com/wp-content/uploads/2024/07/A83_2025-Annual-Report_v5_web.pdf) (Table 16). The study measures detections in sampled U.S. locations; this list is **not** a worldwide sales or ownership ranking.

1. DJI Mini 4 Pro
2. DJI Air 3
3. DJI Mavic 3 Pro
4. DJI Air 2S
5. DJI Air 3S
6. DJI Mavic 3
7. DJI Mavic 3 Classic
8. DJI Mavic 2 Pro
9. DJI Avata 2
10. DJI Mini 3 Pro

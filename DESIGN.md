---
name: "Auto-HR"
description: "A clear recruiting desk for inspecting professional evidence."
colors:
  primary: "#2c58dc"
  primary-hover: "#2147bc"
  navy: "#182838"
  ink: "#162237"
  muted: "#56677f"
  canvas: "#f1f5fa"
  surface: "#fff"
  line: "#e2e7ef"
  confirmed: "#14734d"
  confirmed-bg: "#e1f5eb"
  confirmed-tag-text: "#176444"
  unknown: "#896015"
  unknown-bg: "#fff2d6"
  unknown-tag-text: "#80550e"
  focus: "#79a5ff"
  selected: "#edf3ff"
  danger: "#b33c45"
typography:
  headline:
    fontFamily: "\"Manrope Variable\", Manrope, system-ui, sans-serif"
    fontSize: "34px"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "-0.035em"
  title:
    fontFamily: "\"Manrope Variable\", Manrope, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 760
    lineHeight: 1.45
    letterSpacing: "-0.02em"
  section-title:
    fontFamily: "\"Manrope Variable\", Manrope, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 750
    lineHeight: 1.5
  body:
    fontFamily: "\"Manrope Variable\", Manrope, system-ui, sans-serif"
    fontSize: "14px"
    lineHeight: 1.7
  button:
    fontFamily: "\"Manrope Variable\", Manrope, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 750
    lineHeight: 1.4
  tag:
    fontFamily: "\"Manrope Variable\", Manrope, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 550
    lineHeight: 1.35
rounded:
  tag: "4px"
  field: "5px"
  button: "6px"
  panel: "8px"
  avatar: "9px"
  dialog: "12px"
spacing:
  space-4: "4px"
  space-8: "8px"
  space-10: "10px"
  space-12: "12px"
  space-16: "16px"
  space-18: "18px"
  space-20: "20px"
  space-24: "24px"
  space-26: "26px"
components:
  button-primary:
    rounded: "{rounded.button}"
    padding: "11px 17px"
    typography: "{typography.button}"
    backgroundColor: "{colors.primary}"
    textColor: "{colors.surface}"
  button-secondary:
    rounded: "{rounded.button}"
    padding: "11px 17px"
    typography: "{typography.button}"
    backgroundColor: "{colors.surface}"
    textColor: "#294eaa"
  button-ghost:
    rounded: "{rounded.button}"
    padding: "11px 17px"
    typography: "{typography.button}"
    textColor: "{colors.primary}"
  button-danger:
    rounded: "{rounded.button}"
    padding: "11px 17px"
    typography: "{typography.button}"
    backgroundColor: "{colors.danger}"
    textColor: "{colors.surface}"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
  field:
    backgroundColor: "{colors.surface}"
    textColor: "#233b58"
    rounded: "{rounded.field}"
    padding: "11px 12px"
  panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.panel}"
  tag-confirmed:
    backgroundColor: "{colors.confirmed-bg}"
    textColor: "{colors.confirmed-tag-text}"
    rounded: "{rounded.tag}"
    padding: "4px 8px"
    typography: "{typography.tag}"
  tag-unknown:
    backgroundColor: "{colors.unknown-bg}"
    textColor: "{colors.unknown-tag-text}"
    rounded: "{rounded.tag}"
    padding: "4px 8px"
    typography: "{typography.tag}"
  nav-active:
    backgroundColor: "#294c83"
    textColor: "{colors.surface}"
    rounded: "{rounded.button}"
    padding: "12px 15px"
---

# Design System: Auto-HR

## Overview

**Creative North Star: "The Recruitment Dispatch Desk"**

The Recruitment Dispatch Desk pairs a stable ink-navy frame with cool light reading surfaces. Compact controls and strong Cyrillic headings support repeated scanning; evidence, unknowns, and explicit human decisions carry the visual emphasis.

The system is restrained and operational. Manrope supplies both headings and body text, white panels separate work from the cooler canvas, and small status tags communicate professional evidence without turning the interface into a field of saturated cards.

**Key Characteristics:**
- Stable navy navigation with light reading surfaces.
- One neutral Cyrillic type family with a compact hierarchy.
- Cobalt actions; green confirmed evidence; amber unknowns.
- Flat panels and explicit responsive navigation.

## Colors

Cool neutrals create the reading field; cobalt directs action, while green and amber carry evidence meaning. Frontmatter records the normative values extracted from `frontend/src/styles.css`.

### Primary
- **Action Cobalt:** primary actions, active tabs, links, and the shared action emphasis; the deeper hover token marks interaction.

### Secondary
- **Confirmed Green:** confirmed evidence, with a pale green field and darker dedicated tag text.
- **Unknown Amber:** clarification needs, with a pale amber field and darker dedicated tag text.
- **Destructive Red:** deletion confirmation, kept distinct from evidence state.

### Neutral
- **Rail Navy:** fixed navigation frame.
- **Reading White / Cool Canvas:** panel and page layers respectively.
- **Reading Ink / Muted Slate:** content hierarchy and supporting information.
- **Divider Mist:** thin panel boundaries and separators.
- **Selection Blue / Focus Blue:** selected queue row and visible keyboard focus.

**The Evidence Color Rule.** Use green for confirmed evidence and amber for information that needs clarification; keep the meaning visible in text.

## Typography

**Display Font:** Manrope Variable, Manrope, system-ui, sans-serif.
**Body Font:** The same family. The application bundles the Cyrillic variable font; system-ui is a fallback, not the display identity.

**Character:** Neutral, open, and dense enough for repeated professional inspection. The hierarchy is based on observed roles rather than a mathematical scale.

### Hierarchy
- **Headline:** heavy page heading; responsive sizes descend from the frontmatter value to (29px), (28px), and (27px).
- **Title:** section headings; dossier titles use a heavier (800) weight.
- **Section title:** vacancy/result headings and smaller section anchors.
- **Body:** the root size with paragraph leading; local evidence text and form controls are mostly (11–13px).
- **Label:** tags and supporting metadata use the compact tag role. Primary heading actions increase to (13px) on desktop; ordinary buttons use the button role.
- **Numbers:** ranking values use tabular figures, heavy weight, and tight tracking; they remain visually separate from confirmed coverage.

**The Single Family Rule.** Use Manrope across headings, labels, and reading text; use weight and size to establish hierarchy.

## Layout

The desktop shell has a fixed rail (250px) and a white topbar (64px). Main content is capped at (1800px), centered within the remaining width, with base padding (18px 24px 20px). Repeated gaps use the compact spacing steps recorded above; panel interiors commonly use (16–20px).

The current recruiting workbench uses two columns (1.3fr / 1fr), a (12px) gap, and matching (724px) queue and dossier heights with independent interior scrolling. These are this surface's implemented composition, not mandatory proportions for every future screen. At widths above (1600px), panels become (744px), the gap becomes (16px), and main padding becomes (32px).

At (1350px) and below, the rail contracts to (218px) and the workbench to (1.15fr / 1fr). From (901–1100px), the rail is (200px) and columns are equal. At (900px) and below the rail becomes an overlay drawer, the topbar is (59px), and the workbench shows either the list or the dossier with an explicit back control. Height and interior scrolling constraints release for document scrolling. At (520px) and below, coverage moves into a natural second grid row beneath the candidate identity; form grids become one column, and text fields use (16px) text.

## Elevation & Depth

The workspace is flat by default: white fields, cool canvas, and thin boundaries establish separation. Modal layers use a dark translucent blurred backdrop; only dialogs and the active segmented control carry shadows.

### Shadow Vocabulary
- **Dialog:** `0 24px 80px #0b1c3b38` separates a focused task from the workbench.
- **Active segment:** `0 2px 5px #1b355013` gently lifts the selected white segment.

**The Flat Workbench Rule.** Separate everyday panels with tone and thin borders; reserve the substantial shadow for dialogs.

## Shapes

Compact rounded rectangles define controls, tags, panels, and avatars. Tags use the smallest repeated corner, fields slightly larger, and buttons/navigation share a restrained control corner. Panels use the panel corner; dialogs use the largest reusable corner, reduced to (9px) on narrow screens. Thin solid borders define fields and panels; the upload target alone uses a dashed boundary. User avatars and status dots may be circular; candidate avatars remain rounded rectangles.

## Components

### Buttons

Compact, clear, and label-led. Primary uses cobalt on white text; secondary is white with a blue-gray border and darker blue text; ghost retains cobalt text without a resting fill. Destructive confirmation uses red. Shared padding and shape live in frontmatter. Hover deepens primary or adds a pale field to secondary/ghost. Disabled buttons use opacity (0.55) and a not-allowed cursor. Visible focus is a (3px) focus-blue outline with a (3px) offset. Heading actions have a (44px) minimum height; decision actions have a (42px) minimum height.

### Chips

Small rounded tags wrap rather than truncate. Neutral tags use a pale gray field and border. Confirmed and unknown variants use their dedicated fields and text colors with transparent borders. Text or a check icon identifies meaning independently of hue. Review-state badges are distinct from extracted evidence tags.

### Cards / Containers

White surfaces, thin divider borders, and restrained panel corners. Workbench panels provide their own padded header/content/footer regions rather than applying universal card padding. Result rows use internal separators and comfortable reading space. Selected queue rows use selection blue; hover uses a quieter cool field.

### Inputs / Fields

White fields with a blue-gray stroke (1px), compact field corners, and padding from the field token. Labels are (12px), bold (700), with (8px) separation. Fields use visible focus outlines; the queue search wraps focus with a (2px) focus-blue outline instead. Placeholder text is darker slate for legibility. Textareas resize vertically. Errors pair a pale red field with text; pending forms disable their controls.

### Navigation

The navy rail uses muted pale text, a darker blue active field, and a lighter navy hover field; active and hovered labels turn white. Navigation items have a (45px) minimum height. Outline SVG icons support labels. The mobile drawer overlays a translucent scrim and retains the same navigation vocabulary. Dossier tabs use cobalt text and a (2px) underline for the active tab.

### Evidence dossier

A candidate header, underlined tabs, scrollable evidence region, source-context actions, and a separate decision footer keep inspection and action legible. Evidence quotes remain inspectable; source-context actions can reveal highlighted text. Green confirmed tags and amber unknowns sit beside explanations rather than replacing them. Relative model values use separate muted-blue numeric fields.

### Dialogs and motion

Dialogs use a sticky divided heading, padded form body, constrained viewport height, and internal scrolling. Wide dialogs extend to (830px), ordinary dialogs to (540px). Button state transitions take (150ms); dossier reveal and mobile drawer transitions take (200ms). Loading icons rotate over (1.1s). Reduced-motion preference removes animations and transitions.

## Do's and Don'ts

### Do:
- **Do** keep confirmed coverage separate from relative ranking values.
- **Do** preserve visible keyboard focus and text labels alongside status color.
- **Do** retain the explicit back path when mobile shows an individual dossier.
- **Do** use SVG line icons and the shipped geometric brand mark.

### Don't:
- **Don't** substitute a condensed display face for the neutral Cyrillic hierarchy.
- **Don't** use green and amber as interchangeable decorative accents.
- **Don't** extend dialog shadows to every workbench panel.
- **Don't** replace inspectable evidence with an unexplained score.

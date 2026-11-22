#!/usr/bin/env python3
"""Build video/fairlens-demo.mp4 — the ≤2-minute demo video.

Pipeline: per-scene narration (macOS `say`, offline) → 1280×720 scene frames
(designed HTML cards + REAL dashboard screenshots via #ty&to deep links) →
ffmpeg segments with fades → concat + AAC audio. Every number in the video
comes from the same executed analysis as the write-up.
"""
import json
import re
import os
import subprocess
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = os.path.join(ROOT, "video")
SHOTS = os.path.join(VID, "shots")
os.makedirs(SHOTS, exist_ok=True)
CHROME = os.path.expanduser("~/tools/chrome/browsers/chrome-headless-shell/mac_arm-157.0.8103.78/chrome-headless-shell-mac-arm64/chrome-headless-shell")
FF = os.path.expanduser("~/tools/ffmpeg/node_modules/ffmpeg-static/ffmpeg")

NAVY, CYAN, AMBER, GREY = "#0B1F3A", "#22D3EE", "#F59E0B", "#64748B"

BASE_CSS = f"""
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:1280px;height:720px;overflow:hidden;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif}}
.slide{{width:1280px;height:720px;display:flex;flex-direction:column;position:relative}}
.cap{{background:#0B1F3A;color:#fff;padding:12px 34px;font-size:21px;line-height:1.45;flex:none}}
.cap b{{color:{CYAN}}} .cap .a{{color:{AMBER};font-weight:800}}
.cap .en{{
  position:absolute;left:0;right:0;bottom:0;background:rgba(11,31,58,.94);color:#fff;
  padding:12px 34px;font-size:19px;line-height:1.4;border-top:3px solid {CYAN};
}}
"""

def slide(body):
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{BASE_CSS}</style></head><body><div class='slide'>{body}</div></body></html>"

def head_band(title, sub):
    return f"""<div style="background:linear-gradient(135deg,{NAVY} 0%,#12294d 60%,#0e3a5c 100%);color:#fff;
      padding:20px 40px;border-bottom:4px solid {CYAN}">
      <div style="font-size:30px;font-weight:800;letter-spacing:-.5px">Fair<span style="color:{CYAN}">Lens</span>
      <span style="font-size:14px;font-weight:500;color:#B9C7DA;margin-left:12px">{title}</span></div>
      <div style="font-size:12.5px;color:#D7E1EE;margin-top:4px">{sub}</div></div>"""

def img_stage(src):
    return f"""<div style="flex:1;display:flex;align-items:center;justify-content:center;background:#F1F5F9">
      <img src="{src}" style="max-width:92%;max-height:88%;border-radius:10px;box-shadow:0 10px 40px rgba(11,31,58,.25)"></div>"""

SCENES = [
    dict(
        id="s1", text=("Banks use models to decide who gets a loan. But what if that model quietly approves "
                       "eighty percent of applicants over twenty five, and only sixty two percent of those under? "
                       "This is FairLens — a fairness audit of a loan approval model, built for the Veritas AI hackathon."),
        cap="Banks use models to decide who gets a loan. But what if that model <b>quietly approves 80% of applicants over 25</b> — and only <b>62% of those under</b>?",
        slide=slide(f"""
          <div style="flex:1;background:linear-gradient(135deg,{NAVY} 0%,#12294d 55%,#0e3a5c 100%);color:#fff;
                      display:flex;flex-direction:column;justify-content:center;padding:0 90px">
            <div style="font-size:15px;letter-spacing:.3em;color:{CYAN};font-weight:700">VERITAS AI HACKATHON 2026 · TRACK: AI + FINANCE</div>
            <div style="font-size:74px;font-weight:800;letter-spacing:-2px;margin:14px 0 6px">Fair<span style="color:{CYAN}">Lens</span></div>
            <div style="font-size:24px;color:#D7E1EE;line-height:1.5;max-width:900px">A fairness audit of a loan approval model —
              does it treat <b style="color:{AMBER}">young applicants</b> fairly, and what would fixing it cost?</div>
            <div style="margin-top:36px;display:flex;gap:12px">
              <span style="font-size:13px;background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.5);color:#FFD9A8;padding:6px 14px;border-radius:999px">Prototype — not financial advice</span>
              <span style="font-size:13px;color:#B9C7DA;padding:6px 0">Team: Hao Tan · Statlog German Credit (UCI id=144)</span>
            </div>
          </div>"""),
    ),
    dict(
        id="s2", text=("On one thousand credit applicants, our model is seventy five percent accurate and catches thirty one "
                       "of sixty bad risks. But it approves under twenty fives at a rate twenty seven points lower. Part of that gap "
                       "reflects real risk — young applicants genuinely default more. Part of it does not, and the audit separates the two."),
        cap="Our model: <b>75% accurate</b>, catches <b>31 of 60</b> bad risks. But approval for under-25s is <b>27 points lower</b> — partly real risk, and the audit separates the two.",
        slide=slide(head_band("the audit", "held-out test set · n=200 · seed 42") + img_stage("../../assets/fig2_gap.png"),
        ),
    ),
    dict(
        id="s3", dash="#ty=0.50&to=0.50", shot="dash-default",
        text=("This is the live dashboard, computed on the two hundred held-out applicants. Default policy: the gap is plus twenty seven "
              "points. Total cost, priced one to five, is one hundred sixty five. And it catches only thirty one of sixty bad risks."),
        cap="The live dashboard — 200 held-out applicants. Default 0.5/0.5: <b>gap +27.0 pp</b> · cost <b>165</b> · recall <b>52%</b>.",
    ),
    dict(
        id="s4", dash="#ty=0.25&to=0.23", shot="dash-equal",
        text=("The validation-selected equal-approval policy cuts cost to one twelve, and recall to seventy eight percent. "
              "But its near-zero gap does not transfer to the test split. At thirty young applicants, threshold engineering is "
              "fragile — and reporting that honestly is a finding, not a failure."),
        cap="Equal approval 0.25/0.23 (validation-selected): cost <span class='a'>112 vs 165</span> · recall <span class='a'>78% vs 52%</span> — but the gap does not fully transfer (+23 pp).",
    ),
    dict(
        id="s5", text=("Every policy on one chart. The map is our selection surface; the markers carry test outcomes. "
                       "What survives the transfer: cost and recall. What does not: parity. Choosing the metric — and checking "
                       "the transfer — changes every conclusion."),
        cap="The map says it plainly: <b>cost and recall gains survive the transfer — the parity gap does not</b>. Metric choice changes every conclusion.",
        slide=slide(head_band("the fairness–cost map", "validation selection surface · test outcomes in the labels") + img_stage("../../assets/fig4_frontier.png"),
        ),
    ),
    dict(
        id="s6", text=("And hiding age does not help. Retrained without it, the gap stays at seventeen points. "
                       "Employment history alone carries sixteen percent of age's information. The model rebuilds what you removed."),
        cap="Hiding age fails: gap stays <b>+17.3 pp</b>. <b>Employment history alone carries 16% of age's variance</b> — the model rebuilds what you removed.",
        slide=slide(head_band("fairness through unawareness fails", "remove age → retrain → re-audit") + img_stage("../../assets/fig3_proxy.png"),
        ),
    ),
    dict(
        id="s7", text=("FairLens shows the trade-off you can count. Next: equalised-odds audits, modern data, and user testing "
                       "with real loan officers. This is a prototype, not financial advice. Hao Tan, Veritas AI hackathon."),
        cap="Next: equalized-odds audit · modern data · user tests. <span class='a'>A prototype, not financial advice.</span>",
        slide=slide(f"""
          <div style="flex:1;background:linear-gradient(135deg,{NAVY} 0%,#12294d 55%,#0e3a5c 100%);color:#fff;
                      display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:0 80px">
            <div style="font-size:56px;font-weight:800;letter-spacing:-1px">Fair<span style="color:{CYAN}">Lens</span></div>
            <div style="font-size:21px;color:#D7E1EE;line-height:1.6;max-width:880px;margin-top:14px">
              The trade-off you can <b style="color:{AMBER}">count</b>, not guess.</div>
            <div style="display:flex;gap:14px;margin-top:34px">
              <span style="font-size:14px;background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.45);color:{CYAN};padding:8px 18px;border-radius:999px">Equalized-odds audit →</span>
              <span style="font-size:14px;background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.45);color:{CYAN};padding:8px 18px;border-radius:999px">Modern data (LendingClub) →</span>
              <span style="font-size:14px;background:rgba(34,211,238,.12);border:1px solid rgba(34,211,238,.45);color:{CYAN};padding:8px 18px;border-radius:999px">User tests with loan officers →</span>
            </div>
            <div style="font-size:13px;color:#B9C7DA;margin-top:40px">Hao Tan · Veritas AI International High School Hackathon 2026 ·
            <span style="color:#FFD9A8">prototype — not financial advice</span></div>
          </div>"""),
    ),
]

def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(cmd[0] + " failed:\n" + r.stderr[-1500:])
    return r

def wav_len(path):
    w = wave.open(path)
    return w.getnframes() / w.getframerate()

def shoot(url, out_png):
    run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
         f"--screenshot={out_png}", "--window-size=1280,720",
         "--virtual-time-budget=6000", url])

# ---------- 1. dashboard shots + slide renders ----------
for sc in SCENES:
    if "dash" in sc:
        url = "file://" + os.path.join(ROOT, "dashboard", "index.html") + sc["dash"]
        shoot(url, os.path.join(SHOTS, sc["shot"] + ".png"))
    elif "slide" in sc:
        html_path = os.path.join(SHOTS, sc["id"] + ".html")
        open(html_path, "w").write(sc["slide"])
        shoot("file://" + html_path, os.path.join(SHOTS, sc["id"] + ".png"))
print("frames rendered:", sorted(os.listdir(SHOTS)))

# ---------- 2. narration ----------
total = 0
for sc in SCENES:
    aiff = os.path.join(SHOTS, sc["id"] + ".aiff")
    wav = os.path.join(SHOTS, sc["id"] + ".wav")
    run(["say", "-v", "Samantha", "-r", "180", "-o", aiff, sc["text"]])
    run([FF, "-y", "-i", aiff, "-ar", "44100", "-ac", "2", wav])
    os.remove(aiff)
    sc["dur"] = wav_len(wav) + 0.9          # narration + breathing room
    total += sc["dur"]
    print(f"{sc['id']}: {sc['dur']:.1f}s  ({len(sc['text'].split())} words)")
print(f"total narration video length: {total:.1f}s")

# ---------- 3. per-scene segments (image + fades + narration) ----------
seg_files = []
for sc in SCENES:
    img = os.path.join(SHOTS, sc.get("shot", sc["id"]) + ".png")
    wav = os.path.join(SHOTS, sc["id"] + ".wav")
    seg = os.path.join(SHOTS, sc["id"] + ".mp4")
    d = sc["dur"]
    # caption strip burned at bottom; strip HTML markup, escape drawtext specials.
    # NB: ASCII apostrophes break ffmpeg's filter quoting — use typographic U+2019.
    cap = re.sub(r"<[^>]+>", "", sc["cap"])
    cap = (cap.replace("'", "\u2019")
           .replace(":", r"\:").replace("%", r"\%")
           .replace(",", r"\,"))
    font = "/System/Library/Fonts/HelveticaNeue.ttc"
    # wrap caption to <=72-char lines (drawtext does not wrap); one drawtext per line
    words, lines, cur = cap.split(" "), [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 <= 72 or not cur:
            cur = (cur + " " + w_).strip()
        else:
            lines.append(cur); cur = w_
    lines.append(cur)
    n = len(lines)
    dt = ",".join(
        f"drawtext=text='{ln}':fontfile={font}:fontsize=24:fontcolor=white:"
        f"box=1:boxcolor=0B1F3A@0.92:boxborderw=14:x=(w-text_w)/2:y=h-{n * 34 + 18 - i * 34}:expansion=none"
        for i, ln in enumerate(lines))
    vf = (f"scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0F172A,"
          f"{dt},"
          f"fade=t=in:st=0:d=0.5,fade=t=out:st={d-0.5:.2f}:d=0.5")
    run([FF, "-y", "-loop", "1", "-i", img, "-i", wav,
         "-filter_complex", f"[0:v]{vf}[v]",
         "-map", "[v]", "-map", "1:a",
         "-t", f"{d:.2f}", "-r", "30",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", seg])
    seg_files.append(seg)

# ---------- 4. concat ----------
concat_list = os.path.join(SHOTS, "concat.txt")
with open(concat_list, "w") as f:
    for s in seg_files:
        f.write(f"file '{s}'\n")
out = os.path.join(VID, "fairlens-demo.mp4")
run([FF, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
     "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out])
import re as _re
p = subprocess.run([FF, "-i", out], capture_output=True, text=True)  # ffmpeg-static has no ffprobe
m = _re.search(r"Duration: (\d+):(\d+):([\d.]+)", p.stderr)
dur = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
print(f"FINAL: {out}")
print(f"  duration: {dur:.1f}s | size: {os.path.getsize(out)/1e6:.1f} MB | 1280x720 @30fps")

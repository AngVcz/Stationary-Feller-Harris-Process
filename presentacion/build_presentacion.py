# -*- coding: utf-8 -*-
"""Genera el mapa mental (PNG), la descomposicion de cobertura (PNG) y la
presentacion PowerPoint (deck cientifico ~31 slides) del proyecto SF-Harris.
Narrativa: construir el modelo -> orden de reduccion -> pruebas/graficas/interpretacion
-> por que se puede decir que se reduce (validacion cruzada).
Paleta validada (dataviz skill): colorblind-safe, light mode.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)

# ---------- paleta validada (light) ----------
def C(h): return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
INK="0b0b0b"; INK2="52514e"; MUTED="898781"; GRID="e1e0d9"; SURFACE="fcfcfb"
BLUE="2a78d6"; ORANGE="eb6834"; AQUA="1baf7a"; YELLOW="eda100"
MAGENTA="e87ba4"; GREEN="008300"; VIOLET="4a3aa7"; RED="e34948"
GOOD="0ca30c"; CRIT="d03b3b"

# ============================================================
# MAPA MENTAL
# ============================================================
def make_mindmap():
    fig, ax = plt.subplots(figsize=(16,9), dpi=200)
    ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
    fig.patch.set_facecolor("#"+SURFACE); ax.set_facecolor("#"+SURFACE)

    cx, cy = 8.0, 4.5
    # conectores primero (debajo)
    centers = []
    branches = [
        # (x,y,w,h,color,title,lines)
        (1.0, 6.65, 4.3, 1.95, BLUE,
         "Cadena de Harris  (estado latente)",
         ["stay: copia x_{t-1},  con prob. e^{-α}",
          "jump: dibuja de Q,  con prob. 1-e^{-α}",
          "ACF teórica: ρ(h) = e^{-α h}"]),
        (10.7, 6.65, 4.3, 1.95, ORANGE,
         "Q invariante  (distribución marginal)",
         ["Discreta  ·  GIG  ·  Empírica",
          "marginal estacionaria = Q  (por diseño)",
          "empírica ≈ Historical Simulation"]),
        (0.4, 3.5, 4.1, 1.8, AQUA,
         "Resultado:  P(stay) “decorativa”",
         ["la dinámica temporal (P(stay)) es decorativa",
          "para cobertura (propiedad MARGINAL)",
          "→ 2 ingredientes: Q empírica + Gibbs"]),
        (11.5, 3.5, 4.1, 1.8, VIOLET,
         "α  (persistencia temporal)",
         ["se estima de la ACF:  α = -log(ρ₁)",
          "P(stay) = e^{-α}  gobierna lo conjunto",
          "no afecta la marginal = Q"]),
        (1.0, 0.45, 4.3, 1.95, MAGENTA,
         "Inferencia:  Gibbs sampler",
         ["denoising de τ* (varianza latente)",
          "conjugación: α  y  (μ, σ²)",
          "resuelve la casi-no-identific. del GIG"]),
        (10.7, 0.45, 4.3, 1.95, GREEN,
         "Emisión  (observación)",
         ["r_t | τ_t  ~  N(0, τ_t)",
          "condicionalmente gaussiana",
          "sólo usa τ_t actual"]),
    ]
    for (x,y,w,h,color,title,lines) in branches:
        bx, by = x+w/2, y+h/2
        con = FancyArrowPatch((cx,cy),(bx,by), arrowstyle="-",
            color="#"+GRID, linewidth=2.2, zorder=0)
        ax.add_patch(con)
    # centro
    c = FancyBboxPatch((cx-1.7, cy-0.78), 3.4, 1.56,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#"+INK, edgecolor="#"+INK, linewidth=0, zorder=3)
    ax.add_patch(c)
    ax.text(cx, cy, "SF-Harris", ha="center", va="center", color="white",
        fontsize=30, fontweight="bold", zorder=4)
    ax.text(cx, cy-0.42, "volatilidad estocástica", ha="center", va="center",
        color="#c3c2b7", fontsize=12, zorder=4)
    # cajas
    for (x,y,w,h,color,title,lines) in branches:
        box = FancyBboxPatch((x,y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor="white", edgecolor="#"+color, linewidth=2.6, zorder=2)
        ax.add_patch(box)
        ax.text(x+w/2, y+h-0.30, title, ha="center", va="top", color="#"+color,
            fontsize=14, fontweight="bold", zorder=3)
        for i,ln in enumerate(lines):
            ax.text(x+0.18, y+h-0.78-i*0.42, "• "+ln, ha="left", va="top",
                color="#"+INK2, fontsize=11.5, zorder=3)
    ax.text(8, 8.7, "Mapa mental del modelo SF-Harris", ha="center", va="top",
        color="#"+INK, fontsize=18, fontweight="bold")
    out = os.path.join(ROOT, "mapa_mental.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("mapa mental ->", out)

# ============================================================
# DESCOMPOSICION DE COBERTURA (Test 1)
# ============================================================
def make_coverage_decomposition():
    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200)
    fig.patch.set_facecolor("#"+SURFACE); ax.set_facecolor("#"+SURFACE)
    labels = ["N(0,σ²)\nconstante", "+ τ*\n(adapta var.)",
              "+ Q empírica\n(forma marginal)", "+ Harris\n(dinámica P(stay))"]
    vals = [9.4, 2.0, 2.0, 2.0]
    colors = [RED, ORANGE, AQUA, BLUE]
    bars = ax.bar(labels, vals, color=["#"+c for c in colors], width=0.62,
                  zorder=3, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 11.2)
    ax.set_ylabel("AAD de cobertura (pp)", fontsize=13, color="#"+INK2)
    ax.set_title("Test 1 — Descomposición de cobertura  (IBM, 15 min, dollar bars)",
                fontsize=15, fontweight="bold", color="#"+INK, pad=14)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.18, f"{v:.1f}", ha="center",
                va="bottom", fontsize=13, fontweight="bold", color="#"+INK)
    # salto grande (denoising)
    ax.annotate("", xy=(1, 9.6), xytext=(0, 9.6),
                arrowprops=dict(arrowstyle="-|>", color="#"+GREEN, lw=2.2))
    ax.text(0.5, 9.9, "−7.4 pp  (denoising τ*)", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color="#"+GREEN)
    # saltos nulos
    for x0, x1 in [(1, 2), (2, 3)]:
        ax.annotate("", xy=(x1, 2.45), xytext=(x0, 2.45),
                    arrowprops=dict(arrowstyle="-|>", color="#"+MUTED, lw=1.6))
        ax.text((x0+x1)/2, 2.62, "0.0 pp", ha="center", va="bottom",
                fontsize=11, color="#"+MUTED)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#"+GRID); ax.spines[sp].set_linewidth(1.2)
    ax.tick_params(colors="#"+INK2, labelsize=10.5)
    ax.yaxis.grid(True, color="#"+GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    out = os.path.join(ROOT, "coverage_decomposition.png")
    plt.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print("coverage decomposition ->", out)

# ============================================================
# POWERPOINT
# ============================================================
make_mindmap()
make_coverage_decomposition()
prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height
PAGE = {"n": 0}
TITLE_SHORT = "SF-Harris — de la cadena de Harris a dos ingredientes"

def _noshadow(shape):
    try: shape.shadow.inherit = False
    except Exception: pass

def footer(slide):
    PAGE["n"] += 1
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.35))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = TITLE_SHORT
    r.font.size = Pt(9); r.font.color.rgb = C(MUTED); r.font.name="Calibri"
    p2 = tb.text_frame.add_paragraph()
    # page number right
    tb2 = slide.shapes.add_textbox(Inches(12.5), Inches(7.05), Inches(0.7), Inches(0.35))
    pp = tb2.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.RIGHT
    rr = pp.add_run(); rr.text = str(PAGE["n"]); rr.font.size=Pt(9); rr.font.color.rgb=C(MUTED)

def base_slide(title, accent=BLUE, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(0.14))
    bar.fill.solid(); bar.fill.fore_color.rgb = C(accent); bar.line.fill.background(); _noshadow(bar)
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.30), Inches(12.1), Inches(1.0))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = title
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = C(INK); r.font.name="Calibri"
    if subtitle:
        p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = subtitle
        r2.font.size = Pt(15); r2.font.color.rgb = C(INK2); r2.font.name="Calibri"
    footer(s)
    return s

def bullets(slide, items, left=0.7, top=1.45, width=12.0, height=5.4, size=18, gap=6):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for it in items:
        lvl = it.get("lvl", 0)
        txt = it["text"]
        col = it.get("color", INK)
        bold = it.get("bold", False)
        sz = it.get("size", size - (2 if lvl else 0))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(it.get("gap", gap))
        r = p.add_run(); r.text = ("• " if lvl==0 else "– ") + txt
        r.font.size = Pt(sz); r.font.bold = bold; r.font.color.rgb = C(col); r.font.name="Calibri"
    return tb

def add_table(slide, data, left, top, width, height, col_w=None, header_fill=BLUE,
              header_color="ffffff", body_size=12, header_size=12.5, align_first_left=True):
    rows, cols = len(data), len(data[0])
    gtbl = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = gtbl.table
    if col_w:
        for i,w in enumerate(col_w): tbl.columns[i].width = Inches(w)
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.text = str(data[r][c])
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if (c==0 and align_first_left) else PP_ALIGN.CENTER
            run = para.runs[0]; run.font.name="Calibri"
            if r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = C(header_fill)
                run.font.size = Pt(header_size); run.font.bold = True; run.font.color.rgb = C(header_color)
            else:
                cell.fill.solid(); cell.fill.fore_color.rgb = C(SURFACE) if r%2 else C("f3f2ee")
                run.font.size = Pt(body_size); run.font.color.rgb = C(INK)
            cell.margin_left=Inches(0.08); cell.margin_right=Inches(0.06)
            cell.margin_top=Inches(0.03); cell.margin_bottom=Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return gtbl

def add_image_fit(slide, path, left, top, max_w, max_h, caption=None):
    from PIL import Image
    im = Image.open(path); iw, ih = im.size; ar = iw/ih
    w = max_w; h = w/ar
    if h > max_h: h = max_h; w = h*ar
    l = left + (max_w-w)/2; t = top + (max_h-h)/2
    slide.shapes.add_picture(path, Inches(l), Inches(t), Inches(w), Inches(h))
    if caption:
        cb = slide.shapes.add_textbox(Inches(left), Inches(top+max_h+0.02), Inches(max_w), Inches(0.3))
        p = cb.text_frame.paragraphs[0]; p.alignment=PP_ALIGN.CENTER
        r = p.add_run(); r.text=caption; r.font.size=Pt(11); r.font.color.rgb=C(MUTED); r.font.name="Calibri"

def figpath(name):
    p = os.path.join(REPO, "report", "figures", name)
    return p if os.path.exists(p) else os.path.join(REPO, name)

# ---------- SLIDE 1: Portada ----------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0,0,SW,SH)
bg.fill.solid(); bg.fill.fore_color.rgb = C(INK); bg.line.fill.background(); _noshadow(bg)
acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.5), SW, Inches(0.10))
acc.fill.solid(); acc.fill.fore_color.rgb = C(BLUE); acc.line.fill.background(); _noshadow(acc)
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.65), Inches(11.7), Inches(2.6)); tf=tb.text_frame; tf.word_wrap=True
p=tf.paragraphs[0]; r=p.add_run()
r.text="Modelo SF-Harris: de la cadena de Harris a dos ingredientes"
r.font.size=Pt(36); r.font.bold=True; r.font.color.rgb=C("ffffff"); r.font.name="Calibri"
p2=tf.add_paragraph(); r2=p2.add_run()
r2.text="Construir el modelo → reducirlo → probarlo → justificar la reducción"
r2.font.size=Pt(22); r2.font.color.rgb=C("c3c2b7"); r2.font.name="Calibri"
p3=tf.add_paragraph(); p3.space_before=Pt(16); r3=p3.add_run()
r3.text="Replicación y extensión de Anzarut (2023)"
r3.font.size=Pt(18); r3.font.color.rgb=C(MAGENTA); r3.font.name="Calibri"
tb2 = s.shapes.add_textbox(Inches(0.8), Inches(5.75), Inches(11.7), Inches(1.4)); tf2=tb2.text_frame
p=tf2.paragraphs[0]; r=p.add_run(); r.text="Ángel Vences Adame"; r.font.size=Pt(18); r.font.bold=True; r.font.color.rgb=C("ffffff"); r.font.name="Calibri"
p=tf2.add_paragraph(); r=p.add_run(); r.text="Licenciatura en Actuaría  ·  IIMAS — UNAM  ·  Servicio Social"
r.font.size=Pt(14); r.font.color.rgb=C("c3c2b7"); r.font.name="Calibri"
p=tf2.add_paragraph(); r=p.add_run(); r.text="Asesor: ____________     ·     2025"
r.font.size=Pt(14); r.font.color.rgb=C("c3c2b7"); r.font.name="Calibri"

# ---------- SLIDE 2: La pregunta científica ----------
s = base_slide("La pregunta científica", BLUE, "¿Son necesarios los 3 ingredientes o existe un modelo más simple?")
bullets(s, [
 {"text":"Anzarut (2023) predice volatilidad intradía con un modelo de 3 piezas y reporta cobertura con AAD = 0.8 pp sobre IBM a 15 min.","size":17},
 {"text":"Las 3 piezas del modelo SF-Harris:","bold":True,"color":INK2,"size":17},
 {"text":"Cadena de Harris — estado latente con persistencia P(stay) = e^{-α} (masa puntual) y saltos desde Q.","lvl":1,"size":15},
 {"text":"Q invariante — distribución marginal de la volatilidad (GIG en Anzarut).","lvl":1,"size":15},
 {"text":"Gibbs sampler — inferencia bayesiana y denoising de τ* (varianza latente).","lvl":1,"size":15},
 {"text":"Pregunta central — descompuesta en tres sub-preguntas anclables a evidencia:","bold":True,"color":BLUE,"size":17},
 {"text":"(i) ¿cadena o iid? — ¿la dinámica temporal aporta, o basta la marginal?","lvl":1,"size":15},
 {"text":"(ii) ¿GIG o empírica? — ¿qué forma de Q da mejor cobertura dentro del rango observado?","lvl":1,"size":15},
 {"text":"(iii) ¿P(stay) informativo o andamiaje? — ¿su valor concreto mueve la cobertura, o sólo deja correr el Gibbs?","lvl":1,"size":15},
], top=1.35, height=5.6)

# ---------- SLIDE 3: Fundamento SSM ----------
s = base_slide("Fundamento: del kernel de Markov al SSM", VIOLET)
bullets(s, [
 {"text":"Punto de partida: un kernel de Markov sobre un estado latente (volatilidad). El estado evoluciona y emite observaciones ruidosas.","size":17},
 {"text":"Esto es un modelo de espacio de estados (SSM): transición latente xₜ = log τₜ  +  emisión rₜ | τₜ.","size":17},
 {"text":"Las observaciones rₜ NO son markovianas por sí solas: la dependencia temporal vive en el estado latente.","size":17,"bold":True,"color":INK},
 {"text":"Consecuencia: no se puede estimar la volatilidad de hoy con un promedio simple de rₜ² — hay que filtrar/inferir el estado.","size":16,"color":INK2},
 {"text":"Tres niveles de inferencia en un SSM (este orden es clave para lo que sigue):","bold":True,"color":VIOLET,"size":17},
 {"text":"Filtrado — p(xₜ | r₁:ₜ), el estado hoy dado todo lo observado hasta hoy.","lvl":1,"size":15},
 {"text":"Suavizado — p(xₜ | r₁:T), el estado dados TODOS los datos (pasado y futuro).","lvl":1,"size":15},
 {"text":"Estimación de parámetros — p(θ | r₁:T) con θ = (α, μ, σ²); promedio posterior sobre el ruido de observación.","lvl":1,"size":15},
 {"text":"Aquí el objetivo es el NIVEL 3 (parámetros), no el suavido del estado. Eso decide el método de inferencia (siguiente slide).","size":16,"bold":True,"color":VIOLET},
], top=1.35, height=5.7)

# ---------- SLIDE 4: Por qué Gibbs, no Kalman/PF ----------
s = base_slide("¿Por qué Gibbs y no Kalman o partículas?", VIOLET,
   "El objetivo (parámetros) + la estructura del modelo descartan las alternativas")
bullets(s, [
 {"text":"Kalman: lineal-gaussiano. Aquí la transición (masa puntual + saltos GIG) y la emisión (varianza en el denominador) son no lineales y no gaussianas → Kalman destruye la bimodalidad de la posterior (veremos 16 pp).","size":15},
 {"text":"Particle filter: aproxima la posterior con pesos. Pero el GIG es casi-no-identificable → los pesos degeneran (ESS → 1) en pocas iteraciones.","size":15},
 {"text":"Gibbs: hay conjugación (α y (μ,σ²) tienen priors conjugados) y el objetivo es la posterior sobre parámetros → el promedio posterior resuelve la casi-no-identificabilidad del GIG sin pesos que degeneran.","size":15,"bold":True,"color":VIOLET},
], top=1.3, height=2.9)
fp = figpath("kalman_vs_gibbs_comparison.png")
if os.path.exists(fp):
    add_image_fit(s, fp, 6.6, 1.25, 6.4, 5.6, caption="Kalman 16 pp vs Gibbs 0.2 pp (previsualización del resultado)")
bullets(s, [{"text":"La elección de Gibbs no es un gusto: la siguen el objetivo (parámetros, nivel 3) y la conjugación.","size":14,"color":INK2}],
        left=0.7, top=4.35, width=5.7, height=2.4)

# ---------- SLIDE 5: Ingrediente 1 — Cadena de Harris ----------
s = base_slide("Ingrediente 1 — Cadena de Harris (estado latente)", BLUE)
bullets(s, [
 {"text":"Estado latente xₜ = log τₜ (log-varianza). La cadena salta entre valores:","size":17},
 {"text":"stay: copia xₜ₋₁,  con probabilidad P(stay) = e^{-α}  (masa puntual en el propio estado).","lvl":1,"size":15},
 {"text":"jump: dibuja un nuevo valor de Q,  con probabilidad 1 − e^{-α}.","lvl":1,"size":15},
 {"text":"α es la persistencia temporal: α grande → P(stay) pequeña → muchos saltos → poca memoria.","size":16,"color":INK2},
 {"text":"Anclaje observable de α: la cadena es una cadena de Markov con ACF teórica ρ(h) = e^{-α h}.","bold":True,"color":BLUE,"size":17},
 {"text":"→ se estima α de los datos sin resolver la cadena: α = −log(ρ₁), donde ρ₁ es la primera autocorrelación de log τ (estimada con varianza realizada).","lvl":1,"size":15},
 {"text":"Esto desacopla la estimación de α de la inferencia de los estados: α entra al Gibbs como dato, no como incógnita.",  "size":15,"color":INK2},
], top=1.35, height=5.6)

# ---------- SLIDE 6: Ingrediente 2 — Q invariante ----------
s = base_slide("Ingrediente 2 — Q invariante (distribución marginal)", ORANGE)
bullets(s, [
 {"text":"Q es la distribución invariante de la cadena: la marginal estacionaria de la volatilidad.","size":17},
 {"text":"Tres candidatas: Discreta (soporte finito)  ·  GIG (Anzarut, paramétrica)  ·  Empírica (resamplear datos).","size":17},
 {"text":"Propiedad clave: la marginal = Q POR DISEÑO del kernel, no por un teorema derivado. El kernel está construido para que Q sea invariante.","bold":True,"color":ORANGE,"size":17},
 {"text":"Q empírica ≈ Historical Simulation: la cobertura hereda la forma exacta de la marginal observada.","size":16,"color":INK2},
 {"text":"Precisión importante (evita un error frecuente):","bold":True,"color":RED,"size":16},
 {"text":"La Q empírica se construye con varianza realizada (RV), NO con rₜ².  Empirical(RV) ≠ Empirical(rₜ²).","lvl":1,"size":15,"color":RED},
 {"text":"A diario rₜ² es un estimador ruidoso de la varianza (σ(log RV) ≈ 2.27); por eso la frecuencia importa (Test 3).","lvl":1,"size":15,"color":INK2},
], top=1.35, height=5.6)

# ---------- SLIDE 7: Ingrediente 3 — Gibbs sampler ----------
s = base_slide("Ingrediente 3 — Gibbs sampler (inferencia)", MAGENTA)
bullets(s, [
 {"text":"El Gibbs hace dos cosas a la vez:","bold":True,"color":INK2,"size":17},
 {"text":"Denoising de τ* (varianza latente): reconstruye la varianza “verdadera” promediando el ruido de observación de rₜ².","lvl":1,"size":15},
 {"text":"Estimación de parámetros (α, μ, σ²): posterior sobre θ, no suavido de estados.","lvl":1,"size":15},
 {"text":"Conjugación que lo hace cerrado por bloques:","size":16,"color":INK2},
 {"text":"α  y  (μ, σ²) tienen priors conjugados → muestreo directo, sin pasos de Metropolis.","lvl":1,"size":15},
 {"text":"Nota honesta: el código usa una Gamma (aproximación conjugada), no el Beta exacto para α; documentado como caveat.","lvl":1,"size":13,"color":INK2},
 {"text":"Resuelve la casi-no-identificabilidad del GIG: el promedio posterior estabiliza los parámetros que un particle filter no podría estimar (ESS → 1).","size":16,"bold":True,"color":MAGENTA},
], top=1.35, height=5.6)

# ---------- SLIDE 8: Emisión ----------
s = base_slide("Emisión — la observación", GREEN)
bullets(s, [
 {"text":"rₜ | τₜ  ~  N(0, τₜ): el retorno es gaussiano condicional a la varianza de hoy.","size":18,"bold":True,"color":GREEN},
 {"text":"Condicionalmente gaussiana: la no normalidad marginal de los rendimientos viene de la mezcla sobre τ, no de colas gaussianas pesadas.","size":16,"color":INK2},
 {"text":"Sólo usa τₜ actual: el retorno de hoy depende sólo de la varianza de hoy, no del historial (la memoria está en la cadena, no en la emisión).","size":16,"color":INK2},
 {"text":"Consecuencia para la predicción: el predictivo a un paso es una mezcla de N(0, τ) ponderada por la posterior de τ. La cobertura es una propiedad de esa mezcla.","size":16,"bold":True,"color":INK},
], top=1.4, height=5.6)

# ---------- SLIDE 9: El modelo completo (mapa mental) ----------
s = base_slide("El modelo completo", BLUE, "3 ingredientes + emisión, en un mapa mental")
s.shapes.add_picture(os.path.join(ROOT,"mapa_mental.png"), Inches(0.5), Inches(1.35), Inches(12.3), Inches(5.55))

# ---------- SLIDE 10: Pipeline de inferencia + Tabla 3 ----------
s = base_slide("Pipeline de inferencia y replicación (Tabla 3)", BLUE,
   "datos → dollar bars → saltos/periodicidad → α(ACF) → Gibbs → predictivo")
data = [
 ["Etapa","Qué hace","Salida"],
 ["1. Datos","precios → dollar bars (volumen constante, no tiempo)","retornos rₜ"],
 ["2. Limpieza","detect_and_remove_jumps + estimate_periodicity","serie sin saltos/estacionalidad"],
 ["3. α","α = −log(ρ₁) de la ACF de log RV","persistencia temporal"],
 ["4. Gibbs","denoising de τ* + posterior de (μ,σ²)","posterior de τ"],
 ["5. Predictivo","mezcla N(0,τ) sobre la posterior","intervalo de cobertura"],
]
add_table(s, data, 0.6, 1.5, 7.6, 3.5, col_w=[1.4,4.3,1.9], header_fill=BLUE, body_size=11.5, header_size=12)
bullets(s, [
 {"text":"Replicación (Tabla 3, IBM 15 min, dollar bars):","bold":True,"color":INK2,"size":15},
 {"text":"este trabajo AAD 0.3 pp  ·  Anzarut 0.8 pp  ·  GARCH 3.4 pp.","lvl":1,"size":14,"color":GREEN},
 {"text":"Libre de fuga: saltos + periodicidad ajustados solo sobre train, aplicados al test.","size":13,"color":AQUA},
], left=8.4, top=1.5, width=4.5, height=4.0)

# ---------- SLIDE 11: Estudio de simulación (Tablas II/III) ----------
s = base_slide("Estudio de simulación (Tablas II/III)", VIOLET,
   "validación interna del Gibbs sobre Q continua")
data = [
 ["Método (Q continua)","E_α relativo","Comentario"],
 ["NDNJ / MLE / EM","≈ 1.0×","coinciden: factorización analítica de la verosimilitud"],
 ["Gibbs (denoising de τ*)","≈ 0.5×","el promedio posterior reduce el error de α a la mitad"],
]
add_table(s, data, 0.6, 1.55, 12.1, 1.9, col_w=[4.4,2.2,5.5], header_fill=VIOLET, body_size=13, header_size=13)
bullets(s, [
 {"text":"Con Q continua los tres estimadores (NDNJ, MLE, EM) coinciden: la verosimilitud se factoriza y no hay casi-no-identificabilidad.","size":16},
 {"text":"El Gibbs-a (denoising) reduce el error de estimación de α a la mitad: el denoising no sólo limpia estados, mejora la estimación de parámetros.","size":16,"bold":True,"color":VIOLET},
 {"text":"Caveat de reproducibilidad: el “cap” de α difiere entre corridas (cap=10 vs cap=50) y no es reproducible — pendiente de fijar (slide 28).","size":14,"color":CRIT},
], top=3.75, height=3.2)

# ---------- SLIDE 12: Tesis + orden de reducción (panorama) ----------
s = base_slide("La tesis y el orden de reducción", AQUA,
   "se reduce en 3 pasos, cada uno aislado por una prueba")
bullets(s, [
 {"text":"Tesis: la dinámica temporal (P(stay)) es decorativa para cobertura → el modelo se reduce a 2 ingredientes.","bold":True,"color":AQUA,"size":18},
 {"text":"Orden de reducción (cada paso responde a una sub-pregunta):","bold":True,"color":INK2,"size":17},
 {"text":"① Quitar la dinámica de Harris  —  ¿cadena o iid?  (Tests 1, 2, 10).","lvl":1,"size":15,"color":BLUE},
 {"text":"② Quitar los saltos continuos (GIG → discreta)  —  ¿la forma continua aporta?  (Test 6).","lvl":1,"size":15,"color":ORANGE},
 {"text":"③ Quitar la Q paramétrica (GIG → empírica)  —  ¿GIG o empírica?  (replicación, Test 1).","lvl":1,"size":15,"color":AQUA},
 {"text":"Quedan 2 ingredientes:  Q empírica  +  denoising Gibbs  (estimación de μ,σ, NO suavizado de estados).","bold":True,"color":GREEN,"size":18},
 {"text":"El orden importa: se quita primero lo que es estructuralmente innecesario (dinámica, level-set), y al final lo que sólo gana en colas (GIG).","size":15,"color":INK2},
], top=1.35, height=5.6)

# ---------- SLIDE 13: La razón estructural ----------
s = base_slide("La razón estructural (antes de las pruebas)", AQUA,
   "por qué P(stay) no puede mover la cobertura")
bullets(s, [
 {"text":"La cobertura es una propiedad MARGINAL del predictivo: depende de la distribución de τ, no de su orden temporal.","bold":True,"color":AQUA,"size":18},
 {"text":"P(stay) gobierna lo CONJUNTO (la dependencia temporal, la trayectoria), no la marginal.","size":17,"color":INK2},
 {"text":"Bajo estacionariedad con Q fija, la marginal = Q = empírica, independiente de P(stay).","bold":True,"color":INK,"size":17},
 {"text":"P(stay) sólo reorganiza el ORDEN temporal de los valores (cuánto tiempo se queda uno), no su distribución.","lvl":1,"size":15,"color":INK2},
 {"text":"→ “decorativa” se refiere a la dinámica temporal P(stay), NO a toda la cadena (el Gibbs no es decorativo: es uno de los 2 ingredientes).","bold":True,"color":AQUA,"size":17},
 {"text":"Es un argumento estructural, no un hallazgo empírico: las pruebas (Bloque II/III) lo confirman, no lo fundamentan.","size":15,"color":INK2},
], top=1.35, height=5.6)

# ---------- SLIDE 14: Reducción ① — quitar dinámica Harris ----------
s = base_slide("Reducción ① — quitar la dinámica de Harris", BLUE,
   "Test 1: descomposición de cobertura (IBM 15 min)")
fp = os.path.join(ROOT, "coverage_decomposition.png")
add_image_fit(s, fp, 0.5, 1.4, 7.7, 5.4, caption="9.4 → 2.0 → 2.0 → 2.0 pp")
bullets(s, [
 {"text":"El salto 9.4 → 2.0 pp (−7.4) viene TODO de τ*: adaptar la varianza (saber si hoy es alta o baja).","size":15,"bold":True,"color":INK},
 {"text":"La forma empírica de Q suma 0.0 pp. La dinámica de Harris suma 0.0 pp.","size":15,"color":AQUA},
 {"text":"→ La dinámica temporal no aporta a la cobertura más allá de la marginal.","size":15,"bold":True,"color":BLUE},
 {"text":"Caveat: aad5 = aad1 (el modelo completo iguala al sin-Harris) es parcialmente circular — ambas usan el mismo Gibbs. Por eso se corroborra con Tests 2 y 10.","size":13,"color":CRIT},
], left=8.5, top=1.5, width=4.4, height=5.0)

# ---------- SLIDE 15: Reducción ① — más evidencia ----------
s = base_slide("Reducción ① — más evidencia (Tests 2 y 10)", AQUA)
fp = figpath("gibbs_pstay_sweep.png")
if os.path.exists(fp):
    add_image_fit(s, fp, 6.4, 1.45, 6.6, 4.0, caption="Sweep de P(stay) con Gibbs: AAD ≈ plano (0.41–0.54 pp)")
data = [
 ["Prueba","Qué barre","Hallazgo"],
 ["Test 2 (ε-calib.)","ε en 8 órdenes → P(stay) 0.37–0.98","AAD plano en 0.3 pp"],
 ["Test 10 (con Gibbs)","P(stay) 0.0–0.95 con denoising","AAD plano 0.41–0.54 pp"],
]
add_table(s, data, 0.6, 1.55, 5.6, 2.0, col_w=[1.9,2.3,1.4], header_fill=AQUA, body_size=11, header_size=11.5)
bullets(s, [
 {"text":"Dos barridos independientes (ε y P(stay) directo) coinciden: la cobertura es plana en P(stay) cuando hay denoising.","size":14},
 {"text":"→ Sostiene la reducción ①: la dinámica es decorativa CON denoising.","size":14,"bold":True,"color":AQUA},
 {"text":"Caveat: el sweep plano 0.4 pp es del régimen SIN ruido de observación; el Gibbs real (con ruido) da ~18 pp planos de margen. Cuantitativamente distinto, misma conclusión cualitativa.","size":12.5,"color":CRIT},
], left=0.6, top=3.7, width=5.7, height=3.2)

# ---------- SLIDE 16: Reducción ② — saltos continuos → discreta ----------
s = base_slide("Reducción ② — saltos continuos → discreta", ORANGE,
   "Finding 2 + Test 6 (mezcla doble-exponencial)")
bullets(s, [
 {"text":"La Q continua (GIG) es una mezcla continua de saltos; la Q discreta es un soporte finito. ¿gana algo la continuidad?","size":17},
 {"text":"Finding 2: los saltos continuos no aportan a la cobertura. Test 6 (mezcla doble-exponencial en lugar de GIG) añade 0.0 pp.","bold":True,"color":ORANGE,"size":17},
 {"text":"Interpretación: la cobertura depende de la FORMA de la marginal (colas), no de si el soporte es continuo o discreto. Dentro del rango observado, una discreta fina la reproduce igual.","size":16,"color":INK2},
 {"text":"→ Se puede reducir GIG → discreta sin pérdida de cobertura (en el rango observado).","size":16,"bold":True,"color":ORANGE},
 {"text":"Caveat de honestidad: esto es cierto POR CONSTRUCCIÓN de la descomposición (la discreta calza la empírica por diseño) y no es falsificable con este test. Es una consistencia, no una refutación independiente.","size":14,"color":CRIT},
], top=1.35, height=5.6)

# ---------- SLIDE 17: Reducción ③ — Q paramétrica → empírica ----------
s = base_slide("Reducción ③ — Q paramétrica → empírica", AQUA, "Finding 1: qué forma de Q da mejor cobertura")
data = [
 ["Q","AAD (15 min)","Comentario"],
 ["Empírica (resamplear RV)","0.2–0.3 pp","captura la forma exacta de la marginal"],
 ["Student-t","0.4 pp","colas pesadas paramétricas"],
 ["Gaussiana","0.5 pp","colas demasiado ligeras"],
 ["GIG (Anzarut)","0.8 pp","subestima colas (kurtosis)"],
]
add_table(s, data, 0.6, 1.55, 7.4, 3.0, col_w=[3.0,1.7,2.7], header_fill=AQUA, body_size=12.5, header_size=12.5)
bullets(s, [
 {"text":"Finding 1: la Q empírica supera a todas las paramétricas dentro del rango observado — hereda la forma exacta de la marginal.","bold":True,"color":AQUA,"size":15},
 {"text":"→ Se puede reducir GIG → empírica para cobertura.","size":15,"bold":True,"color":AQUA},
 {"text":"La GIG NO se tira: gana en extrapolación de colas (P99.9+), donde la empírica no puede ir. Único caso donde Q paramétrica vence.","size":14,"color":INK2},
 {"text":"Precisión: el 26 pp del estimador simple (N(0,σ̂²) sin denoising) es otro eje — sin denoising, incluso la empírica se desmorona (slide 18).","size":13,"color":INK2},
], left=8.2, top=1.55, width=4.6, height=4.8)

# ---------- SLIDE 18: Lo que NO se reduce — Gibbs esencial ----------
s = base_slide("Lo que NO se reduce — el Gibbs es esencial", MAGENTA,
   "sin denoising, P(stay) sí importa (y destruye)")
fp = figpath("simple_estimators_comparison.png")
if os.path.exists(fp):
    add_image_fit(s, fp, 0.5, 1.5, 6.4, 4.2, caption="Estimadores simples ≈ 26 pp vs Gibbs 0.2 pp")
fp = figpath("pstay_sweep.png")
if os.path.exists(fp):
    add_image_fit(s, fp, 6.9, 1.5, 6.0, 4.2, caption="Test 9: P(stay) monótonamente destructiva sin denoising")
bullets(s, [
 {"text":"Test 8 (coin-flip): sin denoising, a P(stay)=0.88 el AAD se va a 50 pp — el modelo se vuelve inútil.","size":14,"color":RED},
 {"text":"Estimadores simples (sin Gibbs): ~26 pp. Con Gibbs: 0.2 pp. El denoising es el ingrediente activo.","size":14,"bold":True,"color":MAGENTA},
 {"text":"→ La reducción quita la dinámica, NO el Gibbs. El Gibbs es uno de los 2 ingredientes que sobreviven.","size":14,"bold":True,"color":AQUA},
], left=0.5, top=5.85, width=12.3, height=1.1, size=14)

# ---------- SLIDE 19: El modelo mínimo + por qué fallan alternativas ----------
s = base_slide("El modelo mínimo + por qué fallan las alternativas", GREEN)
bullets(s, [
 {"text":"Modelo mínimo (2 ingredientes):","bold":True,"color":GREEN,"size":18},
 {"text":"τ ~ Empírica(RV)  +  r | τ ~ N(0, τ)  con denoising Gibbs de τ* (estimación de μ,σ, NO suavizado de estados).","lvl":1,"size":15,"bold":True,"color":INK},
 {"text":"Receta en 3 pasos: (1) resamplear RV para Q empírica; (2) Gibbs para la posterior de τ*; (3) mezcla N(0,τ) para el predictivo.","lvl":1,"size":14,"color":INK2},
 {"text":"¿Por qué falla Kalman (16 pp)? Es lineal-gaussiano → colapsa la bimodalidad de la posterior de τ. No puede representar “volatilidad alta hoy sí, hoy no”.","size":15,"color":RED},
 {"text":"¿Por qué fallan los estimadores simples (26 pp)? Promedian el ruido de observación en lugar de separarlo del estado. Sin denoising no hay cobertura.","size":15,"color":RED},
 {"text":"¿Por qué gana la empírica sobre la GIG (0.2 vs 0.8 pp)? La GIG subestima la kurtosis; la empírica la hereda de los datos. Salvo en colas extremas.","size":15,"color":INK2},
], top=1.35, height=4.3)
fp = figpath("kalman_vs_gibbs_comparison.png")
if os.path.exists(fp):
    add_image_fit(s, fp, 7.0, 1.45, 6.0, 5.5, caption="Kalman 16 pp (destruye bimodalidad) vs Gibbs 0.2 pp")

# ---------- SLIDE 20: Las 11 pruebas de ablación ----------
s = base_slide("Las 11 pruebas de ablación (mapa)", VIOLET, "cada prueba aísla un componente")
data = [
 ["#","Prueba","Aísla","Hallazgo"],
 ["1","Descomposición de cobertura","τ* vs Q vs Harris","7.4 pp de τ*, 0.0 pp de Harris"],
 ["2","Calibración de ε","P(stay) vía ε","AAD plano 0.3 pp (P 0.37–0.98)"],
 ["3","Frecuencia","15 min / 1 h / diario","degrada con σ(logRV), no P(stay)"],
 ["4","Activos mexicanos","generalización","diario Hist.Sim 8/8; 1 h SF 5/8"],
 ["5","Detección de crisis","señal direccional","AUC 0.42 (anti-predictivo, v1)"],
 ["6","Mezcla doble-exp","saltos continuos","+0.0 pp (por construcción)"],
 ["7","Stay / jump","señal direccional","AUC 0.51 (azar)"],
 ["8","Coin-flip","denoising off","50 pp a P(stay)=0.88"],
 ["9","Sweep sin Gibbs","P(stay) sin denoising","destructiva: 0.42→50→66 pp"],
 ["10","Sweep con Gibbs","P(stay) con denoising","plano 0.41–0.54 pp"],
 ["K","Kalman vs Gibbs","suavizado lineal","16 pp (destruye bimodalidad)"],
]
add_table(s, data, 0.5, 1.5, 12.3, 5.2, col_w=[0.6,3.1,2.6,6.0], header_fill=VIOLET, body_size=11, header_size=12)

# ---------- SLIDE 21: Prueba 3 (frecuencia) ----------
s = base_slide("Prueba 3 — la frecuencia: el ruido del RV, no P(stay)", MAGENTA)
data = [
 ["Frecuencia","obs/día","σ(log RV)","P(stay)","SF-Harris","Hist. Sim."],
 ["15-min","26","1.04","0.88","0.2 pp","2.0 pp"],
 ["1-h","7","0.87","~0.55","10.3 pp","1.8 pp"],
 ["diario","1","2.27","0.37","2.5 pp","2.0 pp"],
]
add_table(s, data, 0.7, 1.55, 11.9, 2.6, col_w=[2.0,1.7,2.0,1.7,2.3,2.2], header_fill=MAGENTA, body_size=13, header_size=13)
bullets(s, [
 {"text":"El AAD sigue al ruido del estimador de RV (σ), no a P(stay): a 15 min, 26 obs/día → RV bien estimada → σ baja → buen desempeño.","size":16},
 {"text":"A diario, rₜ² es un estimador pésimo de la varianza (σ = 2.27): el Gibbs infla su distribución de saltos.","size":16},
 {"text":"Historical Simulation no necesita estimar RV y elude el problema → gana a diario.","size":16,"color":INK2},
 {"text":"Caveat (provisional): Test 3 conserva la fuga train/test → AADs absolutos provisionales; el signo (AAD sigue a σ, no a P(stay)) es robusto. Ver slide 28.","size":12,"color":CRIT},
], top=4.4, height=2.5)

# ---------- SLIDE 22: Pruebas 5 & 7 (dirección/crisis) ----------
s = base_slide("Pruebas 5 y 7 — no predice dirección ni crisis", RED)
data = [
 ["Señal","AUC","Interpretación"],
 ["Detección de crisis (PIT del modelo, v1)","0.42","anti-predictivo (< 0.5)"],
 ["VIX para detección de crisis","0.84","referencia fuerte"],
 ["Predicción stay/jump (ratio τ*)","0.51","indistinguible de azar"],
 ["P(jump) a diario","0.63 (constante)","prob. incondicional, no señal temporal"],
]
add_table(s, data, 0.7, 1.55, 11.9, 2.7, col_w=[5.3,1.7,4.9], header_fill=RED, body_size=13, header_size=13)
bullets(s, [
 {"text":"El modelo está MENOS sorprendido antes de las crisis que en promedio; P(jump) es una constante, no un pico antes de cambios de régimen.","size":15},
 {"text":"→ SF-Harris no es señal de trading ni detector de crisis. Su valor es la cobertura, no la dirección.","size":15,"bold":True,"color":RED},
 {"text":"Caveat: el AUC 0.42 es del modelo CONGELADO v1 (PIT 0.36 → descalibrado). La versión correcta es v2 rolling (1-paso); el resultado direccional negativo se mantiene, pero el número 0.42 no es el definitivo.","size":13,"color":CRIT},
], top=4.5, height=2.5)

# ---------- SLIDE 23: Prueba 9 (sweep sin Gibbs) ----------
s = base_slide("Prueba 9 — sweep de P(stay) SIN Gibbs", RED,
   "sin denoising, P(stay) es monótonamente destructiva")
fp = figpath("pstay_sweep.png")
if os.path.exists(fp):
    add_image_fit(s, fp, 3.4, 1.5, 8.5, 4.6, caption="AAD 0.42 → 50 → 66 pp al subir P(stay) sin denoising")
bullets(s, [
 {"text":"Sin denoising, cuanto más persistente la cadena (más P(stay)), peor la cobertura: 0.42 → 50 → 66 pp.","size":15,"color":RED},
 {"text":"Esto confirma que P(stay) SÍ importa… sólo cuando falta el Gibbs. Con denoising (Test 10) la curva se aplana.","size":15,"bold":True,"color":AQUA},
 {"text":"→ El Gibbs es el ingrediente que vuelve decorativa a P(stay). Quitar el Gibbs es lo que no se puede hacer.","size":15,"bold":True,"color":MAGENTA},
], left=0.6, top=6.05, width=12.1, height=1.0, size=14)

# ---------- SLIDE 24: Síntesis 3 hallazgos → 2 ingredientes ----------
s = base_slide("Síntesis — 3 hallazgos → 2 ingredientes", AQUA)
data = [
 ["Hallazgo","Qué dice","Prueba(s)","Evidencia"],
 ["1","Q empírica > Q paramétrica","replicación, Test 1","0.2–0.3 vs 0.8 (GIG) pp"],
 ["2","Saltos continuos no aportan","Test 6","+0.0 pp (mezcla doble-exp)"],
 ["3","Dinámica P(stay) decorativa","Tests 1, 2, 9, 10","0.0 pp Harris; AAD plano"],
 ["→","2 ingredientes: Q empírica + denoising Gibbs","—","0.2–0.3 pp (15 min)"],
]
add_table(s, data, 0.6, 1.6, 12.1, 3.0, col_w=[1.0,5.4,3.0,2.7], header_fill=AQUA, body_size=13, header_size=13)
bullets(s, [
 {"text":"Los 3 hallazgos son consistentes entre sí y con la razón estructural (slide 13): la cobertura es marginal, P(stay) es conjunta.","size":15,"color":INK2},
 {"text":"La reducción deja 2 ingredientes: Q empírica (forma de la marginal) + denoising Gibbs (separa ruido de estado).","size":15,"bold":True,"color":GREEN},
], top=4.8, height=2.0)

# ---------- SLIDE 25: Validación 1 — mexicanos ----------
s = base_slide("Validación 1 — activos mexicanos", ORANGE,
   "evidence externa: ¿se generaliza la reducción?")
data = [
 ["Régimen","Resultado","Ganador"],
 ["Diario (8 activos)","Hist.Sim gana 8/8 (SV 8–12 vs Hist 1.4–3.7 pp)","Hist. Sim"],
 ["Intradía 1 h (8 activos)","SF-Harris gana 5/8","SF-Harris"],
]
add_table(s, data, 0.6, 1.55, 12.1, 2.2, col_w=[3.4,6.2,2.5], header_fill=ORANGE, body_size=13, header_size=13)
bullets(s, [
 {"text":"A diario gana Historical Simulation: sin RV intradía, rₜ² es ruidoso (σ≈2.27) y la reducción a “Q empírica” se vuelve indistinguible de Hist.Sim — y más simple.","size":15},
 {"text":"En intradía (1 h) SF-Harris gana 5/8: la Q empírica + denoising aprovechan la RV mejor estimada.","size":15,"color":GREEN},
 {"text":"Caveat: NO es apples-to-apples — SF-Harris es fixed-origin (un origen de expansión) y Hist.Sim es iid (rolling 1-paso), y la emisión gaussiana es “delgada” para activos con colas pesadas. La comparación direcciona, no zanja.","size":13,"color":CRIT},
], top=3.95, height=3.0)

# ---------- SLIDE 26: Validación 2 — selección de modelo ----------
s = base_slide("Validación 2 — selección de modelo (CRPS)", VIOLET,
   "evidence externa: scoring de densidad predictiva")
fp = figpath("model_selection_scoring.png")
if os.path.exists(fp):
    add_image_fit(s, fp, 0.4, 1.45, 7.6, 5.4, caption="CRPS / log-score / PIT / DM")
bullets(s, [
 {"text":"Hist.Sim gana en TODOS los scores (CRPS, log-score, PIT calibrado, DM significativo) en el rango central.","size":15,"bold":True,"color":VIOLET},
 {"text":"El SV (Q paramétrica pesada) SÓLO gana en colas extremas (P99.9+), donde la empírica no tiene datos.","size":15,"color":INK2},
 {"text":"→ Esto apoya la reducción: en el rango donde hay datos, el modelo pesado no supera a la empírica. La GIG se justifica sólo para colas.","size":15,"bold":True,"color":AQUA},
], left=8.1, top=1.55, width=4.9, height=5.0)

# ---------- SLIDE 27: Riesgo VaR/CVaR ----------
s = base_slide("Riesgo — VaR y CVaR", ORANGE,
   "dónde gana la Q empírica y dónde el modelo pesado")
fp = figpath("risk_measurement_backtest.png")
if os.path.exists(fp):
    add_image_fit(s, fp, 0.4, 1.45, 7.6, 5.4, caption="Backtest de VaR/CVaR por nivel")
bullets(s, [
 {"text":"90–95%: Historical Simulation (≈ Q empírica) mejor calibrado — menos violaciones del nivel nominal.","size":15,"color":GREEN},
 {"text":"99%+: el SV (GIG) es más conservador — CVaR 1.3–2.0× mayor, extrapolación hasta 2.69×.","size":15,"color":ORANGE},
 {"text":"→ La reducción no tira la GIG: la Q paramétrica es el ingrediente que sobrevive en colas extremas (P99.9+).","size":15,"bold":True,"color":AQUA},
], left=8.1, top=1.55, width=4.9, height=5.0)

# ---------- SLIDE 28: Limitaciones honestas ----------
s = base_slide("Limitaciones honestas", MUTED, "6 caveats que acotan las afirmaciones")
bullets(s, [
 {"text":"(a) FUGA train/test [residual]: los titulares (0.3 pp dólar, 0.5 pp 15 min, 9.4→2.0, Tabla 3) ya están libres de fuga — saltos y periodicidad se ajustan solo sobre train y se aplican al test. Los Tests 2–10, Kalman y estimadores simples conservan la fuga → sus AADs absolutos son provisionales; las conclusiones relativas (P(stay) decorativa, denoising de τ*) son robustas porque la fuga desplaza ambos brazos por igual.","size":14,"color":ORANGE},
 {"text":"(b) El sweep plano 0.4 pp es del régimen SIN ruido de observación; el Gibbs real (con ruido) da ~18 pp de margen plano. Mismo signo, distinta magnitud.","size":14,"color":CRIT},
 {"text":"(c) “Decorativa” = la dinámica temporal P(stay), NO toda la cadena. El Gibbs es esencial (uno de los 2 ingredientes).","size":14},
 {"text":"(d) El “cap” de α no es reproducible (cap=10 vs cap=50 entre corridas) — pendiente de fijar.","size":14,"color":CRIT},
 {"text":"(e) Test 1 es parcialmente circular (aad5 = aad1 comparten el mismo Gibbs) — por eso se corrobora con Tests 2 y 10.","size":14},
 {"text":"(f) A diario rₜ² es ruidoso (σ≈2.27); el modelo necesita RV intradía. Sin ella, gana Hist.Sim por simplicidad.","size":14},
], top=1.35, height=5.6)

# ---------- SLIDE 29: Lo que ES y NO ES + cuándo ----------
s = base_slide("Lo que ES, lo que NO ES, y cuándo usar cada modelo", BLUE)
data = [
 ["ES (SF-Harris sirve para)","NO ES (no sirve para)"],
 ["Cobertura intradía, 15 min, dollar bars (AAD 0.3 pp)","Predicción direccional (AUC 0.42–0.51)"],
 ["Extrapolación de colas extremas (P99.9+)","Detección de crisis (AUC 0.42)"],
 ["Densidad predictiva calibrada con RV bien estimada","Optimización de portafolios (Sharpe 0.27 vs 0.45)"],
 ["Estimación de parámetros (μ,σ) vía denoising","Suavido de estados (Kalman lo intenta, 16 pp)"],
]
add_table(s, data, 0.6, 1.55, 12.1, 2.7, col_w=[6.05,6.05], header_fill=BLUE, body_size=12.5, header_size=12.5)
bullets(s, [
 {"text":"Cuándo SF-Harris: intradía con RV bien estimada (≥ ~7 obs/día), cobertura y colas extremas.","size":14,"color":GREEN},
 {"text":"Cuándo Historical Simulation: datos diarios o sin RV intradía; simplicidad y mejor calibración central.","size":14,"color":ORANGE},
 {"text":"Cuándo SV paramétrico (GIG): colas extremas donde la empírica no alcanza (P99.9+) y pricing de derivados.","size":14,"color":VIOLET},
], top=4.5, height=2.4)

# ---------- SLIDE 30: Conclusiones + trabajo futuro ----------
s = base_slide("Conclusiones y trabajo futuro", BLUE)
bullets(s, [
 {"text":"3 hallazgos: (1) Q empírica > Q paramétrica en el rango observado; (2) los saltos continuos no aportan; (3) la dinámica P(stay) es decorativa para cobertura.","size":16,"bold":True,"color":AQUA},
 {"text":"El modelo se reduce a 2 ingredientes: Q empírica (≈ Historical Simulation) + denoising bayesiano (Gibbs).","size":16,"bold":True,"color":GREEN},
 {"text":"La decoratividad es estructural: la cobertura es propiedad marginal y P(stay) gobierna lo conjunto — bajo estacionariedad con Q fija.","size":15,"color":INK2},
 {"text":"Futuro: separación train/test estricta y recálculo del AAD (cierra el caveat bloqueante).","size":15,"color":CRIT},
 {"text":"Futuro: particle filter para suavido de estados (nivel 2), cambio de régimen, pricing de opciones, extensión multivariada.","size":15,"color":INK2},
 {"text":"Futuro: comparar SF-Harris y Hist.Sim en el mismo régimen (ambos fixed-origin o ambos rolling 1-paso).","size":15,"color":INK2},
], top=1.4, height=5.6)

# ---------- SLIDE 31: Gracias ----------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0,0,SW,SH)
bg.fill.solid(); bg.fill.fore_color.rgb = C(INK); bg.line.fill.background(); _noshadow(bg)
acc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(3.4), SW, Inches(0.08))
acc.fill.solid(); acc.fill.fore_color.rgb = C(BLUE); acc.line.fill.background(); _noshadow(acc)
tb = s.shapes.add_textbox(Inches(0.8), Inches(3.55), Inches(11.7), Inches(1.5)); tf=tb.text_frame
p=tf.paragraphs[0]; p.alignment=PP_ALIGN.CENTER; r=p.add_run(); r.text="Gracias.  ¿Preguntas?"
r.font.size=Pt(40); r.font.bold=True; r.font.color.rgb=C("ffffff"); r.font.name="Calibri"
p=tf.add_paragraph(); p.alignment=PP_ALIGN.CENTER; r=p.add_run()
r.text="Ángel Vences Adame  ·  IIMAS — UNAM"
r.font.size=Pt(16); r.font.color.rgb=C("c3c2b7"); r.font.name="Calibri"

out = os.path.join(ROOT, "SF-Harris_presentacion.pptx")
try:
    prs.save(out)
except PermissionError:
    # pptx abierto en PowerPoint/visor -> guardar en un nombre alterno
    try:
        out = os.path.join(ROOT, "SF-Harris_presentacion_NEW.pptx")
        prs.save(out)
    except PermissionError:
        out = os.path.join(ROOT, "SF-Harris_presentacion_leakfix.pptx")
        prs.save(out)
    print("AVISO: el pptx destino estaba bloqueado (abierto). Se guardo en:", out)
print("presentacion ->", out)
print("slides:", len(prs.slides._sldIdLst))
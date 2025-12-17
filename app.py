import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from skimage.color import deltaE_ciede2000

# --- Configuração da Página ---
st.set_page_config(
    page_title="PerioMetric",
    page_icon="🐼",
    layout="centered"
)

# --- Gerenciamento de Estado ---
if 'modo_ref' not in st.session_state:
    st.session_state.modo_ref = 'media'
if 'modo_target' not in st.session_state:
    st.session_state.modo_target = 'media'

# --- CSS ---
st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 10px; border: 1px solid #eee; }
    .color-box { height:50px; border-radius:6px; margin-bottom:8px; transition: all 0.3s; }
    .box-inactive { border: 1px solid #ddd; opacity: 0.6; filter: grayscale(0.4); }
    .box-active-blue { border: 3px solid #2196F3; opacity: 1.0; transform: scale(1.02); box-shadow: 0 4px 8px rgba(33, 150, 243, 0.2); }
    .box-active-red { border: 3px solid #ff4b4b; opacity: 1.0; transform: scale(1.02); box-shadow: 0 4px 8px rgba(255, 75, 75, 0.2); }
    div.stButton > button { font-size: 0.8rem; padding: 0.2rem 0.5rem; width: 100%; }
    .diag-box { padding:15px; border-radius:8px; border-left-width: 5px; border-left-style: solid; }
    </style>
    """, unsafe_allow_html=True)

# Inicializa MediaPipe
@st.cache_resource
def get_facemesh():
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# --- Funções Auxiliares ---
def get_lab_color(img_lab, mask):
    mean = cv2.mean(img_lab, mask=mask)[:3]
    return mean[0], mean[1], mean[2]

def get_segment_color(img_lab, mask, percentile=15, mode='dark'):
    masked_pixels = img_lab[mask > 0]
    if len(masked_pixels) == 0: return 0, 0, 0
    sorted_indices = np.argsort(masked_pixels[:, 0])
    limit = int(len(masked_pixels) * (percentile / 100.0))
    limit = max(1, limit)
    if mode == 'dark':
        selected_pixels = masked_pixels[sorted_indices[:limit]]
    else:
        selected_pixels = masked_pixels[sorted_indices[-limit:]]
    mean_val = np.mean(selected_pixels, axis=0)
    return mean_val[0], mean_val[1], mean_val[2]

def lab_to_rgb_display(l, a, b):
    pixel_lab = np.uint8([[[l, a, b]]])
    return cv2.cvtColor(pixel_lab, cv2.COLOR_LAB2RGB)[0][0]

def calculate_ita(l, b):
    if b == 0: b = 0.001
    ita_rad = np.arctan((l - 50) / b)
    return np.degrees(ita_rad)

# --- Processamento ---
def criar_mask_olheira(pts_pixels, image_shape, offset_y):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x_min, x_max = int(np.min(pts_pixels[:,0])), int(np.max(pts_pixels[:,0]))
    y_base = int(np.max(pts_pixels[:,1]))
    pts_poly = np.array([
        [x_min + 4, y_base], [x_max - 4, y_base],           
        [x_max - 4, y_base + offset_y], [x_min + 4, y_base + offset_y]  
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts_poly], 255)
    return mask

def extrair_pontos_pixels(landmarks, indices, width, height):
    pts = []
    for i in indices:
        pt = landmarks[i]
        pts.append([int(pt.x * width), int(pt.y * height)])
    return np.array(pts)

def criar_mask_ref_bochecha(landmarks, w, h):
    cx_l = int(np.mean([landmarks[i].x for i in [205, 50, 123]]) * w)
    cy_l = int(np.mean([landmarks[i].y for i in [205, 50, 123]]) * h)
    cx_r = int(np.mean([landmarks[i].x for i in [425, 280, 352]]) * w)
    cy_r = int(np.mean([landmarks[i].y for i in [425, 280, 352]]) * h)
    mask_circles = np.zeros((h,w), dtype=np.uint8)
    radius = int(h * 0.065)
    cv2.circle(mask_circles, (cx_l, cy_l), radius, 255, -1)
    cv2.circle(mask_circles, (cx_r, cy_r), radius, 255, -1)
    mask_rosto = np.zeros((h, w), dtype=np.uint8)
    points = np.array([[int(l.x * w), int(l.y * h)] for l in landmarks], dtype=np.int32)
    cv2.fillPoly(mask_rosto, [cv2.convexHull(points)], 255)
    return cv2.bitwise_and(mask_circles, mask_rosto)

def processar_imagem(img):
    h, w = img.shape[:2]
    face_mesh = get_facemesh()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks: return None, "Nenhum rosto."
    landmarks = results.multi_face_landmarks[0].landmark
    
    pts_l = extrair_pontos_pixels(landmarks, [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7], w, h)
    pts_r = extrair_pontos_pixels(landmarks, [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382], w, h)
    offset = int(h * 0.045)
    mask_olheiras = cv2.bitwise_or(criar_mask_olheira(pts_l, img.shape, offset), criar_mask_olheira(pts_r, img.shape, offset))
    
    mask_ref_base = criar_mask_ref_bochecha(landmarks, w, h)
    mask_olh_buf = cv2.dilate(mask_olheiras, np.ones((15,15), np.uint8), iterations=1)
    mask_ref_final = cv2.bitwise_and(mask_ref_base, cv2.bitwise_not(mask_olh_buf))
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    lab_ref_avg = get_lab_color(img_lab, mask_ref_final)
    lab_ref_dark = get_segment_color(img_lab, mask_ref_final, percentile=15, mode='dark')
    lab_ref_light = get_segment_color(img_lab, mask_ref_final, percentile=15, mode='light')
    
    lab_olh_avg = get_lab_color(img_lab, mask_olheiras)
    lab_olh_peak = get_segment_color(img_lab, mask_olheiras, percentile=15, mode='dark')
    
    img_vis = img_rgb.copy()
    cv2.drawContours(img_vis, cv2.findContours(mask_olheiras, cv2.RETR_EXTERNAL, 1)[0], -1, (255,0,0), 2)
    cv2.drawContours(img_vis, cv2.findContours(mask_ref_final, cv2.RETR_EXTERNAL, 1)[0], -1, (0,255,0), 2)
    
    return {
        "visual": img_vis,
        "ref": {
            "media": {"lab": lab_ref_avg, "rgb": lab_to_rgb_display(*lab_ref_avg)},
            "escura": {"lab": lab_ref_dark, "rgb": lab_to_rgb_display(*lab_ref_dark)},
            "clara": {"lab": lab_ref_light, "rgb": lab_to_rgb_display(*lab_ref_light)},
        },
        "target": {
            "media": {"lab": lab_olh_avg, "rgb": lab_to_rgb_display(*lab_olh_avg)},
            "pico": {"lab": lab_olh_peak, "rgb": lab_to_rgb_display(*lab_olh_peak)},
        }
    }, None

# --- Interface ---
st.title("🐼 PerioMetric")
uploaded_file = st.file_uploader("Upload da Imagem", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is not None:
        if img.shape[1] > 1000:
            s = 1000/img.shape[1]
            img = cv2.resize(img, (0,0), fx=s, fy=s)
            
        dados, erro = processar_imagem(img)
        
        if erro:
            st.error(erro)
        else:
            st.image(dados["visual"], use_column_width=True)
            st.divider()
            
            # --- SELEÇÃO ---
            st.subheader("🎛️ Parâmetros de Comparação")
            c1, c2, c3 = st.columns(3)
            
            # 1. Ref
            with c1:
                st.markdown("**1. Pele de Referência**")
                rgb_current = dados['ref'][st.session_state.modo_ref]['rgb']
                hex_c = '#%02x%02x%02x' % tuple(rgb_current)
                st.markdown(f'<div class="color-box box-active-blue" style="background-color:{hex_c};"></div>', unsafe_allow_html=True)
                cols = st.columns(3)
                if cols[0].button("Média", key="rm"): st.session_state.modo_ref = 'media'; st.rerun()
                if cols[1].button("Clara", key="rl"): st.session_state.modo_ref = 'clara'; st.rerun()
                if cols[2].button("Escura", key="rd"): st.session_state.modo_ref = 'escura'; st.rerun()

            # 2. Target Média
            with c2:
                st.markdown("**2. Olheira (Alvo)**")
                rgb_m = dados['target']['media']['rgb']
                hex_m = '#%02x%02x%02x' % tuple(rgb_media) if 'rgb_media' in locals() else '#%02x%02x%02x' % tuple(rgb_m)
                cls = "box-active-red" if st.session_state.modo_target == 'media' else "box-inactive"
                st.markdown(f'<div class="color-box {cls}" style="background-color:{hex_m};"></div>', unsafe_allow_html=True)
                if st.button("Média Global", use_container_width=True): st.session_state.modo_target = 'media'; st.rerun()

            # 3. Target Pico
            with c3:
                st.markdown("**Zona Crítica**")
                rgb_p = dados['target']['pico']['rgb']
                hex_p = '#%02x%02x%02x' % tuple(rgb_p)
                cls = "box-active-red" if st.session_state.modo_target == 'pico' else "box-inactive"
                st.markdown(f'<div class="color-box {cls}" style="background-color:{hex_p};"></div>', unsafe_allow_html=True)
                if st.button("Pico Escuro", use_container_width=True): st.session_state.modo_target = 'pico'; st.rerun()

            # =========================================================
            # CÁLCULOS SEPARADOS (DINÂMICO VS PADRÃO)
            # =========================================================
            
            # 1. Dados Dinâmicos (Para Visualização)
            # Obedecem o clique do usuário
            lab_ref_dyn = np.array(dados['ref'][st.session_state.modo_ref]['lab'], dtype=np.float32)
            lab_tar_dyn = np.array(dados['target'][st.session_state.modo_target]['lab'], dtype=np.float32)
            
            dE_visual = deltaE_ciede2000(lab_ref_dyn, lab_tar_dyn)
            
            # ITA dinâmico apenas para exibir na métrica
            ita_ref_dyn = calculate_ita(lab_ref_dyn[0], lab_ref_dyn[2])
            ita_tar_dyn = calculate_ita(lab_tar_dyn[0], lab_tar_dyn[2])
            diff_ita_dyn = ita_ref_dyn - ita_tar_dyn

            # 2. Dados Padrão (Para Diagnóstico Cínico)
            # Sempre Média vs Média (Blindado contra seleção do usuário)
            lab_ref_std = np.array(dados['ref']['media']['lab'], dtype=np.float32)
            lab_tar_std = np.array(dados['target']['media']['lab'], dtype=np.float32)
            
            dE_std = deltaE_ciede2000(lab_ref_std, lab_tar_std)
            
            ita_ref_std = calculate_ita(lab_ref_std[0], lab_ref_std[2])
            ita_tar_std = calculate_ita(lab_tar_std[0], lab_tar_std[2])
            diff_ita_std = ita_ref_std - ita_tar_std

            st.divider()
            
            # MÉTRICAS VISUAIS (DINÂMICAS)
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Diferença Visual (ΔE)", f"{dE_visual:.2f}")
            res_c2.metric("Queda Pigmentar (ITA)", f"{diff_ita_dyn:.2f}°")

            # ----------------------------------------------------
            # 1. CLASSIFICAÇÃO VISUAL (DINÂMICA)
            # Muda conforme o usuário seleciona Média ou Pico
            # ----------------------------------------------------
            st.subheader("👁️ Classificação Visual")
            
            if dE_visual <= 2.3:
                interp_msg = "Imperceptível / Sutil"
                st.success(f"✅ **{interp_msg}**: Diferença mínima (ΔE < 2.3).")
            elif dE_visual <= 10:
                interp_msg = "Perceptível"
                st.info(f"ℹ️ **{interp_msg}**: Diferença estética visível.")
            elif dE_visual <= 20:
                interp_msg = "Clara / Marcada"
                st.warning(f"⚠️ **{interp_msg}**: Contraste óbvio.")
            else:
                interp_msg = "Intensa"
                st.error(f"🚨 **{interp_msg}**: Alto contraste visual.")

            # ----------------------------------------------------
            # 2. DIAGNÓSTICO CAUSAL (ESTÁTICO/PADRÃO)
            # Sempre Média vs Média
            # ----------------------------------------------------
            st.subheader("🩺 Diagnóstico Causal")
            
            if diff_ita_std < 10 and dE_std > 5:
                diag_title = "Sombra Estrutural (Bolsa/Anatomia)"
                diag_desc = "A análise padrão (Média vs Média) indica diferença visual causada por sombra, com pouca alteração de pigmento."
                bg_color, border_color = "#e3f2fd", "#2196F3"
            elif diff_ita_std >= 10:
                diag_title = "Hiperpigmentação (Melanina/Vascular)"
                diag_desc = f"A análise padrão detectou queda real de ITA ({diff_ita_std:.1f}°), indicando pigmentação."
                bg_color, border_color = "#ffebee", "#ef5350"
            else:
                diag_title = "Pele Uniforme"
                diag_desc = "Compatibilidade alta entre as regiões médias."
                bg_color, border_color = "#f1f8e9", "#66bb6a"
                
            st.markdown(f"""
            <div class="diag-box" style="background-color:{bg_color}; border-left-color:{border_color};">
                <h4 style="margin:0; color:#333">{diag_title}</h4>
                <p style="margin:10px 0 0 0; font-size:0.95rem;">{diag_desc}</p>
                <hr style="margin:10px 0; opacity:0.2">
                <p style="font-size:0.8rem; color:#555; margin:0;">
                    🔒 <i>Protocolo Clínico Fixo: <b>Ref Média</b> vs <b>Alvo Média</b> (Independente da seleção visual acima).</i>
                </p>
            </div>
            """, unsafe_allow_html=True)
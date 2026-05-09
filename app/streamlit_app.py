import sys
import streamlit as st
import tempfile
from pathlib import Path

# Add parent directory to path so we can import from inference
sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference.video_pipeline import VideoColorizer

st.set_page_config(page_title="Video Colorization System", page_icon="🎨", layout="wide")

st.title("🎨 Deep Learning-Based Video Colorization System")
st.markdown("Upload a black-and-white video, and the model will add color using a trained U-Net architecture.")

# --- Model Loading Logic ---
# Dynamically find the best weights in the folder
weights_dir = Path("outputs/weights")
fallback_weights = [
    weights_dir / "best_model.pth",
    weights_dir / "interrupted.pth",
]

available_weights = sorted(weights_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True) if weights_dir.exists() else []
default_weights = next((str(path) for path in available_weights + fallback_weights if path.exists()), "")

st.sidebar.title("Settings")
weights_path = st.sidebar.text_input("Model Weights Path", default_weights)

st.sidebar.markdown("---")
st.sidebar.subheader("Post-Processing 🎨")
st.sidebar.info("Use these to fix the tint if the model is still undertrained.")
sat_factor = st.sidebar.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
tint_shift = st.sidebar.slider("Tint (Red ↔ Green)", -50, 50, 0, 1)

if 'colorizer' not in st.session_state or st.session_state.get('loaded_weights') != weights_path:
    with st.spinner("Initializing Attention UNet Model..."):
        try:
            if weights_path and Path(weights_path).is_file():
                st.session_state.colorizer = VideoColorizer(model_path=weights_path)
                st.session_state.loaded_weights = weights_path
                st.sidebar.success(f"Loaded: {Path(weights_path).name}")
            else:
                st.sidebar.warning("No weights found at this path.")
                st.session_state.colorizer = VideoColorizer()
        except RuntimeError:
            st.error("⚠️ Model Architecture Mismatch!")
            st.info("The weights you are trying to load don't match the new Attention UNet architecture. Please use weights generated from your most recent training run (e.g., 'unet_colorization_interrupted.pth').")
            st.stop()

uploaded_file = st.file_uploader("Upload Grayscale Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    st.video(tfile.name)
    
    resolution_option = st.selectbox(
        "Output Resolution (Lower = Faster Processing)", 
        ["480p (Recommended for speed)", "720p", "Original (Very Slow)"],
        index=0
    )
    
    if resolution_option.startswith("480p"):
        max_h = 480
    elif resolution_option.startswith("720p"):
        max_h = 720
    else:
        max_h = None

    col1, col2 = st.columns([1, 2])
    with col1:
        start_btn = st.button("Colorize Video 🪄", use_container_width=True, type="primary")
    with col2:
        st.info("💡 Tip: For best results, use videos with clear lighting.")
        
    if start_btn:
        # Create a unique temporary file for this specific run
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"colorized_{tempfile.gettempprefix()}.mp4")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        # Live Preview Window
        preview_placeholder = st.empty()
        
        def update_progress(current, total, frame=None):
            progress = min(1.0, current / total)
            progress_bar.progress(progress)
            status_text.text(f"Colorizing frame {current} of {total}...")
            # Update the live monitor every few frames
            if frame is not None and current % 5 == 0:
                preview_placeholder.image(frame, channels="BGR", caption="Live AI Processing Preview")
        
        # Process video
        with st.spinner("Colorizing frames..."):
            success = st.session_state.colorizer.process_video(
                tfile.name, 
                output_path, 
                max_height=max_h, 
                progress_callback=update_progress,
                sat_factor=sat_factor,
                tint_shift=tint_shift
            )
        
        preview_placeholder.empty() # Clean up preview after done
            
        if success and Path(output_path).exists():
            status_text.empty()
            progress_bar.empty()
            st.success("✨ Colorization Complete!")
            
            # Use a container for the result
            with st.container():
                with open(output_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                st.markdown("---")
                # Offer download
                st.download_button(
                    label="📥 Download Colorized Video",
                    data=video_bytes,
                    file_name=f"colorized_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary"
                )
            
            # Optional: Clean up input temp file
            try:
                Path(tfile.name).unlink()
            except OSError:
                pass
        else:
            st.error("Failed to process the video. Check if the input file is valid.")
else:
    # Landing page info when no file is uploaded
    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("1. Upload")
        st.write("Choose a B&W video in MP4, AVI, or MOV format.")
    with col_b:
        st.subheader("2. Configure")
        st.write("Adjust saturation and tint in the sidebar for the perfect look.")
    with col_c:
        st.subheader("3. Colorize")
        st.write("Watch as the AI restores color frame by frame!")

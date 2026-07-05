function xaiApp() {
  return {
    imageFile: null,
    fileName: '',
    dragging: false,
    loading: false,
    error: '',
    result: null,
    selectedMethods: ['gradcam', 'gradcam_pp', 'lime', 'shap', 'ig'],
    allMethods: [
      { id: 'gradcam',     label: 'Grad-CAM' },
      { id: 'gradcam_pp',  label: 'Grad-CAM++' },
      { id: 'lime',        label: 'LIME' },
      { id: 'shap',        label: 'SHAP' },
      { id: 'ig',          label: 'Integrated Gradients' },
    ],

    init() {
      // Check health
      fetch('/api/health').catch(() => {});
    },

    handleFile(file) {
      if (!file) return;
      
      // On some Windows systems, file.type can be empty. Check extension as fallback.
      const isImage = file.type.startsWith('image/') || 
                      file.name.toLowerCase().match(/\.(jpg|jpeg|png|gif|bmp|webp)$/);
                      
      if (!isImage) {
        this.error = 'Please upload an image file.';
        return;
      }
      this.imageFile = file;
      this.fileName = file.name;
      this.error = '';
      this.result = null;
    },

    handleDrop(event) {
      this.dragging = false;
      const file = event.dataTransfer?.files?.[0];
      this.handleFile(file);
    },

    async analyze() {
      if (!this.imageFile || this.selectedMethods.length === 0) return;
      this.loading = true;
      this.error = '';

      const fd = new FormData();
      fd.append('file', this.imageFile);

      try {
        const resp = await fetch(
          '/api/explain?methods=' + this.selectedMethods.join(','),
          { method: 'POST', body: fd }
        );
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ detail: resp.statusText }));
          throw new Error(err.detail || 'Server error');
        }
        this.result = await resp.json();
      } catch (e) {
        this.error = e.message || 'Analysis failed. Check that the server is running and the model is loaded.';
      } finally {
        this.loading = false;
      }
    },
  };
}
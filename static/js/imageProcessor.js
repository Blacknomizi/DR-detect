self.addEventListener('message', function(e) {
    const { imageData } = e.data;

    const formData = new FormData();
    formData.append('image', imageData, 'frame.png');

    fetch('/functions/processEachImg', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
    .then(data => {
        self.postMessage({ processedImageData: data });
    })
    .catch(error => {
        self.postMessage({ error: error.message });
    });
});
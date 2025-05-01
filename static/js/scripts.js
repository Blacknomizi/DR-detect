function captureScreen() {
    navigator.mediaDevices.getDisplayMedia({ video: true })
    .then(function(stream) {
        window.opener.postMessage(stream, "*");
    })
    .catch(error => {
        console.error('Error capturing the screen: ', error);
    });
}

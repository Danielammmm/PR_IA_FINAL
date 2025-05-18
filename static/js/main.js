const video = document.getElementById("video");
const textDisplay = document.getElementById("textDisplay");

async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

async function captureFrame() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("image", blob);

        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();
        if (result.letter) {
            textDisplay.textContent += result.letter;
        }
    }, "image/jpeg");
}

video.addEventListener("loadeddata", () => {
    setInterval(captureFrame, 1000);  // Capturar cada segundo
});

startCamera();

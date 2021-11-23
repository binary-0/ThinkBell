var canvas = document.getElementById("arCanvas");
var remoteStream;

const imageSrc = ["img/firejook.png", "img/cloud.png"];

const streamConstraints = { audio: false, video: true };
const mtcnnForwardParams = {
    // limiting the search space to larger faces for webcam detection
    minFaceSize: 200
}

//positions for sunglasess
var results = []

//utility functions
async function getFace(localVideo, options){
    results = await faceapi.mtcnn(localVideo, options)
}

faceapi.loadMtcnnModel('/weights')
faceapi.loadFaceRecognitionModel('/weights')

function arBroadcast(imageNum){
    navigator.mediaDevices.getUserMedia(streamConstraints).then(function (stream) {
        let localVideo = document.createElement("video")
        localVideo.srcObject = stream;
        localVideo.autoplay = true
        localVideo.addEventListener('playing', () => {
            let ctx = canvas.getContext("2d");
            let image = new Image()
            image.src = imageSrc[imageNum];
            // image.src = "img/sunglasses.png"
                
            function step() {
                getFace(localVideo, mtcnnForwardParams)
                ctx.drawImage(localVideo, 0, 0)
                results.map(result => {
                    ctx.drawImage(
                        image,
                        result.faceDetection.box.x - 100,
                        result.faceDetection.box.y - 180,
                        result.faceDetection.box.width * 2,
                        result.faceDetection.box.width * (image.height / image.width) * 2
                    )
                })
                requestAnimationFrame(step)
            }
    
            requestAnimationFrame(step)
        })
    
        localStream = canvas.captureStream(30)
    }).catch(function (err) {
        console.log('An error ocurred when accessing media devices', err);
    });
}

function imageScreen(){
    if(canvas.style.display == 'none'){
        canvas.style.display = 'block';
        arBroadcast(0);
    }

    else{
        canvas.style.display = 'none';
    }
}

function image2Screen(){
    if(canvas.style.display == 'none'){
        canvas.style.display = 'block';
        arBroadcast(1);
    }

    else{
        canvas.style.display = 'none';
    }
    // navigator.mediaDevices.getUserMedia(streamConstraints).then(function (stream) {
    //     let localVideo = document.createElement("video")
    //     localVideo.srcObject = stream;
    //     localVideo.autoplay = true
    //     localVideo.addEventListener('playing', () => {
    //         let ctx = canvas.getContext("2d");
    //         let image = new Image()
    //         image.src = "img/cloud.png"
                
    //         function step() {
    //             getFace(localVideo, mtcnnForwardParams)
    //             ctx.drawImage(localVideo, 0, 0)
    //             results.map(result => {
    //                 ctx.drawImage(
    //                     image,
    //                     result.faceDetection.box.x + 15,
    //                     result.faceDetection.box.y + 30,
    //                     result.faceDetection.box.width,
    //                     result.faceDetection.box.width * (image.height / image.width)
    //                 )
    //             })
    //             requestAnimationFrame(step)
    //         }
    
    //         requestAnimationFrame(step)
    //     })
    
    //     localStream = canvas.captureStream(30)
    // }).catch(function (err) {
    //     console.log('An error ocurred when accessing media devices', err);
    // });
}
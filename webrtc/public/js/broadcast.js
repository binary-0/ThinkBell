let engLog = [];

function getStatus() {

    console.log("get try");
    setInterval(sendStatus, 5000);

    // // var output = {sender:"sender", receiver:"receiver"
    // //     ,command:"chat", type:"text", data:"msg"};
    // socket.emit("message", content);
    socket.on('chat', message=>{
        engLog.push(message);
        console.log('Received from ',message.name, ", engagement : ", message.eng);
    })
}

// setInterval(sendStatus, 5000);

function sendStatus() {

    if(engagementResult!=null){
        console.log("send try");

        let toSend = new Object();
        toSend.name = document.getElementById('studentName').value;
        toSend.eng = engagementResult;
        
        socket.emit("message", toSend);
    }

}

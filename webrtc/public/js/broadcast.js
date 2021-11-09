
function getStatus() {

    console.log("get try");
    // // var output = {sender:"sender", receiver:"receiver"
    // //     ,command:"chat", type:"text", data:"msg"};
    // socket.emit("message", content);
    socket.on('chat', message=>{
        console.log('Received : ', message);
    })
}

function sendStatus() {

    console.log("send try");

    socket.emit("message", engagementResult);

}

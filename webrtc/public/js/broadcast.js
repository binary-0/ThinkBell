let engLog = [];
let names = [];
let start_lecture = 0;

function getStatus() {
   

    console.log("get try");
    setInterval(sendStatus, 1000);
    document.getElementById("preform").style.display ="none";
    // // var output = {sender:"sender", receiver:"receiver"
    // //     ,command:"chat", type:"text", data:"msg"};
    // socket.emit("message", content);
    socket.on('chat', message=>{
        if(start_lecture === 0)
        {
            engLog.push(message);
            names.push(message.name);            
            start_lecture = 1;
            console.log("first person");
        }
        else if(names.indexOf(message.name) < 0) //다른 사람의 정보가 들어올 때
        {
            names.push(message.name);
            engLog.push(message);
            console.log("another person");
        }
        else if(names.indexOf(message.name) >= 0) //같은 사람의 정보가 들어올 때
        {
            engLog[names.indexOf(message.name)] = message;
            console.log("same person");
        }                   
    })
}
// setInterval(sendStatus, 5000);
function sendStatus() {

    if(engagementResult!=null){
        console.log("send try");

        let toSend = new Object();
        let agent = JSON.stringify(student_agent_count);
        let badge = JSON.stringify(stu1_today_badge);
        let gauge = JSON.stringify(engagement_gauge[0]);
        let score = JSON.stringify(ranking_score[0]);
        let handpose = JSON.stringify(objects["handGestureStatus"]);
        toSend.name = document.getElementById('studentName').value;        
        toSend.agent = agent;
        toSend.badge = badge;
        toSend.gauge = gauge;
        toSend.score = score;
        toSend.handpose = handpose;
        socket.emit("message", toSend);
    }
}


let agentmessage = document.getElementById('agentmessage');
let sidebadgebar = document.getElementById('sidebadgebar');
let agentmessagebutton = document.getElementById('agentmessagebutton');
let badgebutton = document.getElementById('badgebutton');
let std1healthbar = document.getElementById('std1healthbar');
let std2healthbar = document.getElementById('std2healthbar');
let std3healthbar = document.getElementById('std3healthbar');
let std4healthbar = document.getElementById('std4healthbar');
let changehealthbar1 = document.getElementById('changehealthbar1');
let changehealthbar2 = document.getElementById('changehealthbar2');
let changehealthbar3 = document.getElementById('changehealthbar3');
let changehealthbar4 = document.getElementById('changehealthbar4');
let stdname1 = document.getElementById('stdname1');
let stdname2 = document.getElementById('stdname2');
let stdname3 = document.getElementById('stdname3');
let stdname4 = document.getElementById('stdname4');
let changehealthbarword1 = document.getElementById('changehealthbarword1');
let changehealthbarword2 = document.getElementById('changehealthbarword2');
let changehealthbarword3 = document.getElementById('changehealthbarword3');
let changehealthbarword4 = document.getElementById('changehealthbarword4');
let handok1 = document.getElementById('handok1');
let handraise1 = document.getElementById('handraise1');
let handok2 = document.getElementById('handok2');
let handraise2 = document.getElementById('handraise2');
let handok3 = document.getElementById('handok3');
let handraise3 = document.getElementById('handraise3');
let handok4 = document.getElementById('handok4');
let handraise4 = document.getElementById('handraise4');
/*성완 코드 넣기*/ 

agentmessagebutton.addEventListener('click',function(){
    sidebadgebar.style.display = "none";
})

badgebutton.addEventListener('click',function(){
    sidebadgebar.style.display = "flex";
})

function hidehealthbar(){
    std1healthbar.style.display = "none";
    std2healthbar.style.display = "none";
    std3healthbar.style.display = "none";
    std4healthbar.style.display = "none";
}

function showhealthbar(){
    std1healthbar.style.display = "block";
    std2healthbar.style.display = "block";
    std3healthbar.style.display = "block";
    std4healthbar.style.display = "block";
}

$('#gaugecheckbox').click(function(){
    var checked = $('#gaugecheckbox').is(':checked');
       
    if(checked){
        showhealthbar();  
    }
    else{
        hidehealthbar();
    }
});

function agentposturemaker(){
    let agentposture = document.createElement('div');
    agentposture.className = 'agentmessage';
    let agentposture1 = document.createElement('div');
    agentposture1.className = 'postureimage';
    let agentposture2 = document.createElement('div');
    agentposture2.className = 'bar2';
    let agentposture3 = document.createElement('div');
    agentposture3.className = 'txt';
    agentposture3.innerHTML="자리로 돌아와 주세요!";
    agentposture.appendChild(agentposture1);
    agentposture.appendChild(agentposture2);
    agentposture.appendChild(agentposture3);
    agentmessage.appendChild(agentposture);
}

function agentsleepmaker(){
    let agentsleep = document.createElement('div');
    agentsleep.className = 'agentmessage';
    let agentsleep1 = document.createElement('div');
    agentsleep1.className = 'sleepimage';
    let agentsleep2 = document.createElement('div');
    agentsleep2.className = 'bar1';
    let agentsleep3 = document.createElement('div');
    agentsleep3.className = 'txt';
    agentsleep3.innerHTML="졸음에서 깨어보아요!";
    agentsleep.appendChild(agentsleep1);
    agentsleep.appendChild(agentsleep2);
    agentsleep.appendChild(agentsleep3);

    agentmessage.appendChild(agentsleep);
}


function agentspeekmaker(){ // sleep , speek 구분하기
    let agentspeek = document.createElement('div');
    agentspeek.className = 'agentmessage';
    let agentspeek1 = document.createElement('div');
    agentspeek1.className = 'speekimage';
    let agentspeek2 = document.createElement('div');
    agentspeek2.className = 'bar1';
    let agentspeek3 = document.createElement('div');
    agentspeek3.className = 'txt';
    agentspeek3.innerHTML="발표를 해볼까요?";
    agentspeek.appendChild(agentspeek1);
    agentspeek.appendChild(agentspeek2);
    agentspeek.appendChild(agentspeek3);

    agentmessage.appendChild(agentspeek);
}


function agentfocusmaker(){
    let agentfocus = document.createElement('div');
    agentfocus.className = 'agentmessage';
    let agentfocus1 = document.createElement('div');
    agentfocus1.className = 'fireimage';
    let agentfocus2 = document.createElement('div');
    agentfocus2.className = 'bar1';
    let agentfocus3 = document.createElement('div');
    agentfocus3.className = 'txt';
    agentfocus3.innerHTML="수업에 더 집중해 볼까요?";
    agentfocus.appendChild(agentfocus1);
    agentfocus.appendChild(agentfocus2);
    agentfocus.appendChild(agentfocus3);
    agentmessage.appendChild(agentfocus);
}

/*발표상*/
function speekbadgemaker(name){
    let speekbadge = document.createElement('div');
    speekbadge.className = 'speekbadgemessage';
    let speekbadge1 = document.createElement('div');
    speekbadge1.className = 'message';
    let speekbadgename = document.createElement('span');
    speekbadgename.innerHTML = String(name)+' 학생이&nbsp;'
    let speekbadge2 = document.createElement('span');
    speekbadge2.className = 'redmessage';
    speekbadge2.innerHTML = '발표상'
    let speekbadge3 = document.createElement('span');
    speekbadge3.innerHTML = '을 획득했어요!'
    speekbadge1.appendChild(speekbadgename);
    speekbadge1.appendChild(speekbadge2);
    speekbadge1.appendChild(speekbadge3);
    speekbadge.appendChild(speekbadge1);
    sidebadgebar.appendChild(speekbadge);
}

/*노력상*/
function focusbadgemaker(name){
    let focusbadge = document.createElement('div');
    focusbadge.className = 'firebadgemessage';
    let focusbadge1 = document.createElement('div');
    focusbadge1.className = 'message';
    let focusbadgename = document.createElement('span');
    focusbadgename.innerHTML = String(name)+' 학생이&nbsp;'
    let focusbadge2 = document.createElement('span');
    focusbadge2.className = 'yellowmessage';
    focusbadge2.innerHTML = '노력상'
    let focusbadge3 = document.createElement('span');
    focusbadge3.innerHTML = '을 획득했어요!'
    focusbadge1.appendChild(focusbadgename);
    focusbadge1.appendChild(focusbadge2);
    focusbadge1.appendChild(focusbadge3);
    focusbadge.appendChild(focusbadge1);
    sidebadgebar.appendChild(focusbadge);
}

/*몰입상*/
function eyebadgemaker(name){
    let eyebadge = document.createElement('div');
    eyebadge.className = 'eyebadgemessage';
    let eyebadge1 = document.createElement('div');
    eyebadge1.className = 'message';
    let eyebadgename = document.createElement('span');
    eyebadgename.innerHTML = String(name)+' 학생이&nbsp;'
    let eyebadge2 = document.createElement('span');
    eyebadge2.className = 'greenmessage';
    eyebadge2.innerHTML = '몰입상'
    let eyebadge3 = document.createElement('span');
    eyebadge3.innerHTML = '을 획득했어요!'
    eyebadge1.appendChild(eyebadgename);
    eyebadge1.appendChild(eyebadge2);
    eyebadge1.appendChild(eyebadge3);
    eyebadge.appendChild(eyebadge1);
    sidebadgebar.appendChild(eyebadge);
}

/*의지상*/
function willbadgemaker(name){
    let willbadge = document.createElement('div');
    willbadge.className = 'fistbadgemessage';
    let willbadge1 = document.createElement('div');
    willbadge1.className = 'message';
    let willbadgename = document.createElement('span');
    willbadgename.innerHTML = String(name)+' 학생이&nbsp;'
    let willbadge2 = document.createElement('span');
    willbadge2.className = 'bluemessage';
    willbadge2.innerHTML = '의지상'
    let willbadge3 = document.createElement('span');
    willbadge3.innerHTML = '을 획득했어요!'
    willbadge1.appendChild(willbadgename);
    willbadge1.appendChild(willbadge2);
    willbadge1.appendChild(willbadge3);
    willbadge.appendChild(willbadge1);
    sidebadgebar.appendChild(willbadge);
}

/*몰입도 게이지 변경 방법*/
function changescore(score,num){
    if(num ===1){ 
        changehealthbar1.style.width = String(score)+"%";
        if(score<=40){
            changehealthbar1.style.background = "#FF5037";
            changehealthbarword1.style.color = "#FFA56C";
        }
        else if(score<=70){
            changehealthbar1.style.background = "#FFD151";
            changehealthbarword1.style.color = "#EE7B33";
        }
        else{
            changehealthbar1.style.background = "#40CC30";
            changehealthbarword1.style.color = "white"
        }
        changehealthbarword1.innerHTML = "몰입도 "+String(score)+" %";
    }
    else if(num ===2){
        changehealthbar2.style.width = String(score)+"%";
        if(score<=40){
            changehealthbar2.style.background = "#FF5037";
            changehealthbarword2.style.color = "#FFA56C";
        }
        else if(score<=70){
            changehealthbar2.style.background = "#FFD151";
            changehealthbarword2.style.color = "#EE7B33";
        }
        else{
            changehealthbar2.style.background = "#40CC30";
            changehealthbarword2.style.color = "white"
        }
        changehealthbarword2.innerHTML = "몰입도 "+String(score)+" %";
    }
    else if(num ===3){
        changehealthbar3.style.width = String(score)+"%";
        if(score<=40){
            changehealthbar3.style.background = "#FF5037";
            changehealthbarword3.style.color = "#FFA56C";
        }
        else if(score<=70){
            changehealthbar3.style.background = "#FFD151";
            changehealthbarword3.style.color = "#EE7B33";
        }
        else{
            changehealthbar3.style.background = "#40CC30";
            changehealthbarword3.style.color = "white"
        }
        changehealthbarword3.innerHTML = "몰입도 "+String(score)+" %";
    }
    else if(num ===4){
        changehealthbar4.style.width = String(score)+"%";
        if(score<=40){
            changehealthbar4.style.background = "#FF5037";
            changehealthbarword4.style.color = "#FFA56C";
        }
        else if(score<=70){
            changehealthbar4.style.background = "#FFD151";
            changehealthbarword4.style.color = "#EE7B33";
        }
        else{
            changehealthbar4.style.background = "#40CC30";
            changehealthbarword4.style.color = "white"
        }
        changehealthbarword4.innerHTML = "몰입도 "+String(score)+" %";
    }
}

// 이름 변경
function changename(name,num){
    // if(num === 1)
    // stdname1.innerHTML = name; 첫번째 이름은 = "나"
    if(num === 2)
    stdname2.innerHTML = name;
    else if(num === 3)
    stdname3.innerHTML = name;
    else if(num === 4)
    stdname4.innerHTML = name;
}

function handGesture(gesture,num){
    
    if(num === 1){
        if(gesture==5){
            //raise
            handraise1.style.display = "block";
            setTimeout(() => {handraise1.style.display = "none";}, 5000);
        }
        else if(gesture==3){
            //ok
            handok1.style.display = "block";
            setTimeout(() => {handok1.style.display = "none";}, 5000);
        }
    }
    else if(num ===2){
        if(gesture==5){
            //raise
            handraise2.style.display = "block";
            setTimeout(() => {handraise2.style.display = "none";}, 5000);
        }
        else if(gesture==3){
            //ok
            handok2.style.display = "block";
            setTimeout(() => {handok2.style.display = "none";}, 5000);
        }
    }
    else if(num ===3){
        if(gesture==5){
            //raise
            handraise3.style.display = "block";
            setTimeout(() => {handraise3.style.display = "none";}, 5000);
        }
        else if(gesture==3){
            //ok
            handok3.style.display = "block";
            setTimeout(() => {handok3.style.display = "none";}, 5000);
        }
    }
    else if(num ===4){
        if(gesture==5){
            //raise
            handraise4.style.display = "block";
            setTimeout(() => {handraise4.style.display = "none";}, 5000);
        }
        else if(gesture==3){
            //ok
            handok4.style.display = "block";
            setTimeout(() => {handok4.style.display = "none";}, 5000);
        }
    }
}
"use strict";

function callbackMove(mv){ // called each time a head movement is detected
  if (PACMAN){
    PACMAN.update_mv(mv);
  }
}

// entry point:
function pacmanAR(){
  pacmanStart(headControlStart);
}

function pacmanStart(callback){
  const el = document.getElementById("pacman");
  if (Modernizr.canvas && Modernizr.localstorage && 
    Modernizr.audio && (Modernizr.audio.ogg || Modernizr.audio.mp3)) {
    window.setTimeout(function () { 
      PACMAN.init(el, "./");
      callback();
    }, 0);
  } else { 
    el.innerHTML = "Sorry, needs a decent browser<br /><small>" + 
    "(firefox 3.6+, Chrome 4+, Opera 10+ and Safari 4+)</small>";
  }
}

function headControlStart(){
  HeadControls.init({
    settings: {
      tol: {
        rx: 1,//do not move if head turn more than this value (in degrees) from head rest position
        ry: 3,
        s: 5 //do not move forward/backward if head is larger/smaller than this percent from the rest position
      },
      sensibility: {
        rx: 1.5,
        ry: 1,
        s: 1
      }
    },
    canvasId: 'headControlsCanvas',
    callbackMove: callbackMove,
    callbackReady: function(errCode){
      if (errCode){
        switch(errCode){
          case 'WEBCAM_UNAVAILABLE':
            alert('Cannot found or use the webcam. You should accept to share the webcam bro otherwise we cannot detect your face !');
            break;
        }
        console.log('ERROR: HEAD CONTROLS NOT READY. errCode =', errCode);
      } else {
        console.log('INFO: HEAD CONTROLS ARE READY :)');
        HeadControls.toggle(true);
        const domStartButton = document.getElementById('start');
        const domResumeButton = document.getElementById('resume');
        domStartButton.style.display = 'inline-block';
        domStartButton.onclick = function(){
          if (PACMAN){
            HeadControls.reset_restHeadPosition();
            PACMAN.start();
            domStartButton.style.display = 'none';
            domResumeButton.style.display = 'inline-block';
          }
        }
        domResumeButton.onclick = function(){
          if (PACMAN){
            HeadControls.reset_restHeadPosition();
            PACMAN.resume();
            // domStartButton.style.display = 'none';
          }
        }
      }
    },
    NNCPath: '../neuralNets/', // where to find NN_DEFAULT.json from the current path
    animateDelay: 2 // avoid DOM lags
  }); //end HeadControls.init params
}

function pacmanScreen(){
  const screen = document.getElementById('gameDiv');
  const notice = document.getElementById('gaming');
  const text = document.getElementById('pacman_text');
  const icon = document.getElementById('pacman_picture');
  const mobicon = document.getElementById('pacman_mob_picture');
  const gameCam = document.getElementById('headControlsCanvas');
  text.innerHTML = '팩맨 Agent 진행 중';
  if(screen.style.display == 'none'){
    screen.style.display = 'block';
    notice.style.display = 'block';
    text.style.display = 'inline-block';
    icon.style.display = 'inline-block';
    gameCam.style.display = 'inline-block';
    var audio = new Audio('../pacman/audio/intermission.mp3');
    audio.volume = 1.0;
    audio.play();
  }
  else{
    screen.style.display = 'none'; 
    notice.style.display = 'none';
    text.style.display = 'none';
    icon.style.display = 'none';
    mobicon.style.display = 'none';
    gameCam.style.display = 'none';
  }
}

function gameON(){
  var audio = new Audio('../pacman/audio/intermission.mp3');
  audio.volume = 1.0;
  audio.play();
  const screen = document.getElementById('gameDiv');
  const notice = document.getElementById('gaming');
  const text = document.getElementById('pacman_text');
  const icon = document.getElementById('pacman_picture');
  const mobicon = document.getElementById('pacman_mob_picture');
  const gameCam = document.getElementById('headControlsCanvas');
  text.innerHTML = '팩맨 Agent 진행 중';
  if(screen.style.display == 'none'){
    notice.style.display = 'block';
    screen.style.display = 'block';
    text.style.display = 'inline-block';
    icon.style.display = 'inline-block';
    mobicon.style.display = 'none';
    gameCam.style.display = 'inline-block';
  }
}

function clearFail(){
  const video = document.getElementById('localVideo');
  const notice = document.getElementById('gaming');
  const failtext = document.getElementById('pacman_text');
  const mobicon = document.getElementById('pacman_mob_picture');
  const icon = document.getElementById('pacman_picture');
  notice.style.display = 'block';
  icon.style.display = 'inline-block';
  mobicon.style.display = 'inline-block';
  failtext.innerHTML = '팩맨 Agent 실패!';
  failtext.style.display = 'inline-block';
  setTimeout(() => {
    notice.style.display = 'none';
    failtext.style.display = 'none';
    icon.style.display = 'none';
    mobicon.style.display = 'none';
  }, 5000);
}
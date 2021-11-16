let present_btn = 1;
    let modal1 = 0;
    let modal2 =0;
    document.getElementById("modal1_on").onclick = function () {
        if(modal1 === 0)
        {
            document.getElementById("modal1").style.display = "flex";
            modal1 = 1;
        }
        else
        {
            document.getElementById("modal1").style.display = "none";
            modal1 = 0;
        }      
    };
    document.getElementById("modal2_on").onclick = function () {
        if(modal2 === 0)
        {
            document.getElementById("modal2").style.display = "flex";
            modal2 = 1;
        }
        else
        {
            document.getElementById("modal2").style.display = "none";
            modal2 = 0;
        }   
    };

    function dp_menu(){
            let click = document.getElementById("drop1");
            if(click.style.display === "none"){
                click.style.display = "block";
 
            }else{
                click.style.display = "none";
            }
        } 

    $("#my_btn1").click(function(){
        $("#mypage_background" + String(present_btn)).hide();
        $("#mypage_background1").show();
        $("#my_btn" + String(present_btn)).css("background-color","#8DD8D1");
        $("#my_btn1").css("background-color","#13A99B");
        present_btn = 1;
    });

    $("#my_btn2").click(function(){
        $("#mypage_background" + String(present_btn)).hide();
        $("#mypage_background2").show();
        $("#my_btn" + String(present_btn)).css("background-color","#8DD8D1");
        $("#my_btn2").css("background-color","#13A99B");
        present_btn = 2;
    }); 
    
    $("#my_btn3").click(function(){
        $("#mypage_background" + String(present_btn)).hide();
        $("#mypage_background3").show();
        $("#my_btn" + String(present_btn)).css("background-color","#8DD8D1");
        $("#my_btn3").css("background-color","#13A99B");
        present_btn = 3;
    }); 

    // 원형 그래프
    var bar1 = document.querySelector('#bar1');
    var bar_value1 = document.querySelector('#bar_value1');                 

function progress(per) {
  var progress = per / 100;
  var dashoffset = (2 * Math.PI *54) * (1 - progress);
  bar_value1.innerHTML= per +'%';
  if(per < 30)
  {
    $("#bar1").css("stroke","#EB9872");
    
  }
  else if(per <60)
  {
    $("#bar1").css("stroke"," #FFEA2C");    
  }
  else if(per <= 100)
  {
    $("#bar1").css("stroke"," #00CA08");
  }  
  bar1.style.strokeDashoffset = dashoffset;
  
}
function max_log(eng,neu,not)
{
    if(eng > neu && eng > not)
    {
        return 2;
    }
    else if(neu > eng && neu > not)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


        let agent1_on = [0,0,0,0];
        let agent1_count = [0,0,0,0];
        let agent1_length = [0,0,0,0];
        let stu1_agent_count = [0,0,0,0];
        let stu1_color = [0,0,0];
        let scoreTemp1;
        let stu1Score;
        let totalStamp = 0;
        let log_update = [0,0,0];           
        let engage_again =0;
        let lecture_time = 120; //수업 시간은 2분이라고 가정
        let engagement_gauge1 = 50;
        let stu1_badge2 = 0;
        let stu1_badge3 = 0;
        let stu1_badge3_time = 0;

        setInterval(function(){           
            console.log("objects_color : " + objects["colorStat"]);
            console.log("objects_generalStat0 : " + objects["generalStat"][0]);
            console.log("objects_generalStat1 : " + objects["generalStat"][1]);
            console.log("objects_generalStat2 : " + objects["generalStat"][2]);
            console.log("objects_generalStat3 : " + objects["generalStat"][3]);
            console.log("objects_handStat : " + objects["handStat"]);            
            if(objects["colorStat"] === "2" ) //Engagement 발생
            {
                stu1_color[0]++;
                log_update[0]++;
                engagement_gauge1 = engagement_gauge1 + (1 - engagement_gauge1/100);
                               
            }
            else if(objects["colorStat"] === "1") //Neutral 발생
            {
                stu1_color[1]++;
                log_update[1]++;
            }
            else //Not engagement 발생
            {
                stu1_color[2]++;
                log_update[2]++;
                engagement_gauge1 = engagement_gauge1 - (engagement_gauge1/100);
            }
            console.log(engagement_gauge1);
            totalStamp+=1;
            stu1Score = (stu1_color[0] * 2 + stu1_color[1]) / (totalStamp * 2 ) * 100; //점수 계산           
            stu1Score = Math.round(stu1Score); 
            progress(stu1Score); //원형 그래프 업데이트
            bar1.style.strokeDasharray = (2 * Math.PI *54) ;

            //자리비움 발생
            if(objects["generalStat"][0] === "1")
            {
                agent1_on[0]+=1;                
            }               
            else
            {
                
              /*if(agent1_count[0] == 2)
                {
                     //자리비움이 유지될 경우 Ai-agent 발생 로그에 저장
                    let agentboard = document.getElementById("agentbar1");
                    let myboard = document.getElementById("mypage_bar1");
                    let pbar = document.createElement("div");
                    let mybar = document.createElement("div");
                    mybar.className ="progress-agent1-noborder";                
                    mybar.style.width =  String(agent1_length[0]/2.5) + "%";
                    pbar.className ="progress-agent1-noborder";                
                    pbar.style.width = String(agent1_length[0]/2.5) + "%";
                    agentboard.appendChild(pbar);  
                    myboard.appendChild(mybar);              
                }*/
                agent1_on[0] = 0;
                agent1_count[0] = 0;  
                let pboard = document.getElementById("agentbar1");
                let mypage = document.getElementById("mypage_bar1");
                let pbar = document.createElement("div");
                let mybar = document.createElement("div");
                mybar.className ="progress-bar-0";                
                mybar.style.width = "0.25%";
                pbar.className ="progress-bar-0";                
                pbar.style.width = "0.25%";                
                pboard.appendChild(pbar);       
                mypage.appendChild(mybar);
            }
            //자세불량 발생
            if(objects["generalStat"][1] === "1")
            {
                agent1_on[1]+=1;                            
            }               
            else
            {
                agent1_on[1] = 0;
                agent1_count[1] = 0;        
                let pboard = document.getElementById("agentbar2");
                let mypage = document.getElementById("mypage_bar2");
                let pbar = document.createElement("div");
                let mybar = document.createElement("div");
                mybar.className ="progress-bar-0";                
                mybar.style.width = "0.25%";
                pbar.className ="progress-bar-0";                
                pbar.style.width = "0.25%";
                pboard.appendChild(pbar);   
                mypage.appendChild(mybar);
            }
            //졸음 발생
            if(objects["generalStat"][3] === "1")
            {
                agent1_on[2]+=1;                                 
            }               
            else
            {
                agent1_on[2] = 0;
                agent1_count[2] = 0;   
                 let pboard = document.getElementById("agentbar3");
                 let mypage = document.getElementById("mypage_bar3");
                let pbar = document.createElement("div");
                let mybar = document.createElement("div");
                mybar.className ="progress-bar-0";                
                mybar.style.width = "0.25%";
                pbar.className ="progress-bar-0";                
                pbar.style.width = "0.25%";
                pboard.appendChild(pbar);  
                mypage.appendChild(mybar);
                //만약 수업의 20% 이상 졸았다가 다시 그 두배 이상 집중 했을 경우
                if(engage_again > lecture_time * (4/10))
                {
                    $("#today_badge1").show();
                }
            }

            //자리비움 기록
            if(agent1_on[0] >= 5 ) //횟수 기록 기준
            {  
                //학생 몰입도 점수 페이지 progressbar 연장
                if(agent1_count[0] === 0) {
                stu1_agent_count[0]++;
                agent1_count[0] = 1;
                document.getElementById("progress_agent1").style ="width : " + String(stu1_agent_count[0]*10)+"%";
                $("#agent_text1").text(String(stu1_agent_count[0])+"회");
                }
                /*
                if(agent1_on[0] >= 5 ) //로그 기록 기준
                {
                    agent1_length[0] = agent1_on[0];
                    agent1_count[0] = 2;
                }*/   
                let agentboard = document.getElementById("agentbar1");
                    let myboard = document.getElementById("mypage_bar1");
                    let pbar = document.createElement("div");
                    let mybar = document.createElement("div");
                    mybar.className ="progress-agent1-noborder";                
                    mybar.style.width =  "0.5%";
                    pbar.className ="progress-agent1-noborder";                
                    pbar.style.width = "0.5%";
                    agentboard.appendChild(pbar);  
                    myboard.appendChild(mybar);                                 
            } 
                         
            //자세불량 기록
            if(agent1_on[1] >= 5)
            {   
                if(agent1_count[1] === 0){                
                stu1_agent_count[1]++;
                agent1_count[1] = 1;
                document.getElementById("progress_agent2").style ="width : " + String(stu1_agent_count[1]*10)+"%";
                $("#agent_text2").text(String(stu1_agent_count[1])+"회");
                }
                let agentboard = document.getElementById("agentbar2");                
                let myboard = document.getElementById("mypage_bar2");
                let pbar = document.createElement("div");
                let mybar = document.createElement("div");
                mybar.className ="progress-agent2-noborder";                
                mybar.style.width = "0.5%";
                pbar.className ="progress-agent2-noborder";                
                pbar.style.width = "0.5%";
                agentboard.appendChild(pbar);
                myboard.appendChild(mybar); 
            } 
            
            //졸음 기록
            if(agent1_on[2] >= 5)
            {   
                if(agent1_count[2] === 0){
                stu1_agent_count[2]++;
                agent1_count[2] = 1;
                document.getElementById("progress_agent3").style ="width : " + String(stu1_agent_count[2]*10)+"%";
                $("#agent_text3").text(String(stu1_agent_count[2])+"회");
                }
                let agentboard = document.getElementById("agentbar3");
                let myboard = document.getElementById("mypage_bar3");
                let pbar = document.createElement("div");
                let mybar = document.createElement("div");
                mybar.className ="progress-agent3-noborder";                
                mybar.style.width = "0.5%";
                pbar.className ="progress-agent3-noborder";                
                pbar.style.width = "0.5%";
                agentboard.appendChild(pbar);
                myboard.appendChild(mybar);
            } 
            //시작할때 progressbar 초기화
            if(stu1_agent_count[0] === 0)
            {
                $(".progress-border_log").css("border-radius","5px")
            }
            //자리비움 말고 다른 agent 발생시 자리비움 초기화
            if(stu1_agent_count[0] === 0 && (stu1_agent_count[1] > 0 || stu1_agent_count[2] > 0))
            {
                $("#progress_agent1").css("width", "0%");
            }
            if(totalStamp % 30 === 0)
            {
                let log_board = document.getElementById("all_log");    
                let mypage_board = document.getElementById("mypage_totalbar");
                let maxlog = max_log(log_update[0],log_update[1],log_update[2]);
                let pbar = document.createElement("div");
                let mybar = document.createElement("div");
                if(maxlog === 2) //Engagement가 제일 많을 때
                {
                    pbar.className = "progress-bar-green";
                    pbar.style = "width : 10%";
                    mybar.className = "progress-bar-green";
                    mybar.style = "width : 10%";
                }
                else if(maxlog === 1) //Neutral이 제일 많을 때
                {
                    pbar.className = "progress-bar-yellow";
                    pbar.style = "width : 10%";
                    mybar.className = "progress-bar-yellow";
                    mybar.style = "width : 10%";
                }
                else if(maxlog === 0) //Not engagement가 제일 많을 때
                {
                    pbar.className = "progress-bar-red";
                    pbar.style = "width : 10%";
                    mybar.className = "progress-bar-red";
                    mybar.style = "width : 10%";
                }
                log_board.appendChild(pbar);
                mypage_board.appendChild(mybar);
                log_update.fill(0);                
            }
            //배지시스템 -> 수업 시간은 2분이라고 가정
            //수업 시간의 70% 이상 engagement -> 몰입상 
            if(stu1_color[0] > lecture_time * (7/10))
            {
                $("#today_badge4").show();
            }
            //수업에 20% 이상 졸음 감지되고 다시 회복될 때 -> 의지상
            if(agent1_on[2] > lecture_time * (2/10))            
            {
               stu1_badge2 = agent1_on[2];                
            }
            else if(stu1_badge2 > 0)
            {
                engage_again++;
            }
            else if(agent1_on[2] > 0) //만약 중간에 또 졸았을 경우
            {
                stu1_badge2 = 0;
            }

            //집중도 게이지가 낮을 경우 -> 노력상 
            if(engagement_gauge1 < 30)
            {
                stu1_badge3++;    
                                 
            }
            //수업 시간의 10% 이상 유지되었을 경우
            else if(stu1_badge3 > lecture_time * (1/10) && engagement_gauge1 > 70)
            {
                stu1_badge3_time++;
                if(stu1_badge3_time > stu1_badge3*2)
                {
                    $("#today_badge2").show();
                }
            }     
           
            

        },1000); 
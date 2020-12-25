import {predict} from './predict.js';


let demoSentences = [
    "I luv apples ! ",
    "The weather was so nice! Yesterday I went to beach.", 
    "This chemical is widly used in the swimming pools market. Chlorine is well known for its sanatizing properties."
];

function shuffle(array) {
    var currentIndex = array.length, temporaryValue, randomIndex;
  
    // While there remain elements to shuffle...
    while (0 !== currentIndex) {
  
      // Pick a remaining element...
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex -= 1;
  
      // And swap it with the current element.
      temporaryValue = array[currentIndex];
      array[currentIndex] = array[randomIndex];
      array[randomIndex] = temporaryValue;
    }
  
    return array;
}

let availableExamples = shuffle(demoSentences);


function launch_demo(){
  if (demoIsVisible){
    $("#tutorial-line").animate({ opacity: 0 }, 1000);
    demoIsVisible = false;
  }
  let demoSentence = availableExamples.pop();
  availableExamples.unshift(demoSentence);
  let speed = 40;
  $('#text-box-input').html("").blur();
  typewrite("text-box-input", demoSentence, false, speed);
  let wait = speed*demoSentence.length+500;
  setTimeout(() => {
    predict()
  }, wait);

  return;
}

function typewrite (target, text, loop, speed) {
    // (A) SET DEFAULT OPTIONS
    target = document.getElementById(target);
    if (speed === undefined) { speed = 200; }
    if (loop === undefined) { loop = false; }
   
    // (B) DRAW TYPEWRITER
    let pointer = 0;
    let timer = setInterval(function(){
      pointer++;
      if (pointer <= text.length) {
        target.innerHTML = text.substring(0, pointer);
      } else {
        if (loop) { pointer = 0; }
        else { clearInterval(timer); }
      }
    }, speed);
}



// EVENT LISTENERS 
//
// Keybinding: 17: ctrl, 68: D
var down = {17:false, 68:false};

$(document).keydown(function(e) {
  if (e.keyCode in down) {
      down[e.keyCode] = true;
      if (down[17] && down[68]) {
        e.preventDefault();
        launch_demo();
      }
  }
}).keyup(function(e) {
  if (e.keyCode in down) {
      down[e.keyCode] = false;
  }
});

// On click on the demo button
$("#demo-button").on("click", launch_demo);

// When something is typed, remove the demo area
var demoIsVisible = true;

document.getElementById("text-box-input").addEventListener("input", function() {
  if(demoIsVisible && $("#text-box-input").text() !=""){
    $("#tutorial-line").animate({ opacity: 0 }, 1000);
    demoIsVisible = false;
  }
  else if($("#text-box-input").text() ==="" && !demoIsVisible){
    $("#tutorial-line").animate({ opacity: 1 });
    demoIsVisible = true;
  }
}, false);
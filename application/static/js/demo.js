import predict from "./predict";


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
    demoSentence = availableExamples.pop();
    availableExamples.unshift(demoSentence);
    $('#text-box-input').html("");
    $('#text-box-input').focus();
    typewrite("text-box-input", demoSentence, speed=300, loop=false);
    $('#text-box-input').focus();
    setTimeout(() => {
        predict()
    }, 500);

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


// 17: ctrl, 68: D
var down = {17:false, 68:false};

$("document").keydown(function(e) {
    down[e.keyCode] = true;
}).keyup(function(e) {
    if (down[17] && down[68]) {
        e.preventDefault();
        launch_demo();
    }
    down[e.keyCode] = false;
});​
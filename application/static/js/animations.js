export var animateButton = function() {
    let e = $("#arrow-button");
    //reset animation
    e.removeClass( "animate" );
    
    e.addClass('animate');
    setTimeout(function(){
        e.removeClass( "animate" );
    },1000);
  };
  
  

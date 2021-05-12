export var animateButton = function() {
    let e = $("#arrow-button");
    //reset animation
    e.removeClass( "is-clicked-demo" );
    
    e.addClass('is-clicked-demo');
    setTimeout(function(){
        e.removeClass( "is-clicked-demo" );
    },700);
  };
  
  

export var animateDemoButton = function() {
    let e = $("#demo-button");
    //reset animation
    e.removeClass( "is-clicked-demo" );
    
    e.addClass('is-clicked-demo');
    setTimeout(function(){
        e.removeClass( "is-clicked-demo" );
    },100);
  };
  
  

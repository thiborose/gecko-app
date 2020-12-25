export var animateButton = function() {
    let e = $("#arrow-button");
    //reset animation
    e.removeClass( "is-clicked" );
    
    e.addClass('is-clicked');
    setTimeout(function(){
        e.removeClass( "is-clicked" );
    },700);
  };
  
  

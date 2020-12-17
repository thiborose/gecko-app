import {predict} from './predict.js';

$('#text-box-input').keypress(function (event){
    // If enter is pressed and shift is not pressed
    if (event.keyCode == 13 && !event.shiftKey) {
        event.preventDefault();
        // Un-focusing from the text box
        $('#text-box-input').blur()
        predict();
    }
});


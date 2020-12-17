// removes formatting on paste

$('#text-box-input').on('paste', function(e) {
  e.preventDefault();
  var current_text = $("#text-box-input").text();
  var cb_text = '';
  if (e.clipboardData || e.originalEvent.clipboardData) {
    cb_text = (e.originalEvent || e).clipboardData.getData('text/plain');
  } else if (window.clipboardData) {
    cb_text = window.clipboardData.getData('Text');
  }

  var new_text = current_text + cb_text;

  $("#text-box-input").html(new_text);
  $("#text-box-input").blur();

});


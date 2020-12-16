// removes formatting on paste

$('#text-box-input').on('paste', function(e) {
  e.preventDefault();
  var text = '';
  if (e.clipboardData || e.originalEvent.clipboardData) {
    text = (e.originalEvent || e).clipboardData.getData('text/plain');
  } else if (window.clipboardData) {
    text = window.clipboardData.getData('Text');
  }
  $("#text-box-input").html(text);
});
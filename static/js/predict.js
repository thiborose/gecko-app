$("#arrow-button").click(function(){
    var text = $("#textarea-input").text();
    
    if (text !== ''){

        $.ajax({
            url: "/predict",
            type: "get",
            data: {jsdata: text},
            success: function(response) {
                $("#textarea-output").html(response);
            },
            error: function(xhr) {
                $("#textarea-output").html("Gecko seems to be tired today...  (︶︹︶)");
            }
        });
        $("#textarea-output").html("Loading...");

    }

    
});
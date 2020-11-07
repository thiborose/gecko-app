$("#arrow-button").click(function(){
    var text = $("#text-box-input").text();
    
    if (text !== ''){

        $.ajax({
            url: "/predict",
            type: "get",
            data: {jsdata: text},
            success: function(response) {
                $("#text-box-output").html(response);
            },
            error: function(xhr) {
                $("#text-box-output").html("Gecko seems to be tired today...  (︶︹︶)");
            }
        });
        $("#text-box-output").html("Loading...");

    }

    
});
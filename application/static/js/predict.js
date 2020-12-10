$("#arrow-button").click(function(){
    // Inserting predicted text in the right box
    var text = $("#text-box-input").text();
    
    if (text !== ''){

        $.ajax({
            url: "/predict",
            type: "get",
            // dataType: 'json',
            data: {jsdata: text},
            success: function(response) {
                //let tags = JSON.parse(response);

                // Inserting predicted text with highlights in the right box
                $("#text-box-output").html(response["output"]);

                // Highlighting deletion changes in the left box
                $("#text-box-input").html(response["input"]);

            },
            error: function(xhr) {
                $("#text-box-output").html("Gecko seems to be tired today...  (︶︹︶)");
            }
        });
        $("#text-box-output").html("Loading...");

    }

    
    
});
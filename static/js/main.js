$(document).ready(function() {
    'use strict';
    $('.summary').hide();

    //$('input[name=cluster_select]').removeAttr('checked');
    $('.select-button').on('change', function() {
        if ($(this).hasClass('active')) {
            $(this).removeClass('active');
        } else {
            $(this).addClass('active');
        }

    });
    $('.view-button').click(function() {
        $('.view-button').removeClass('active');
        $(this).addClass('active');

        // Clear all cluster selection checkboxes.
        $('input[name=cluster_select]').removeAttr('checked');

    });
    $('.view-button').on('change', function() {
        $('#sg-form').submit();
    });
    $('.page-button').on('click', function() {
        $('#page-form').submit();
    });

    $('.document').mouseover(function() {
        $(this).find(".summary").show();
    });

    $('.document').mouseout(function() {
        $(this).find(".summary").hide();
    });

    $('.search-bar').click(function() {
        $(this).removeClass("shadow-depth-1");
        $(this).addClass("shadow-depth-2");
        $(this).addClass("clicked");
    });

    $('.search-bar').mouseover(function() {
        $(this).removeClass("shadow-depth-1");
        $(this).addClass("shadow-depth-2");
    });

    $('.search-bar').mouseout(function() {
        if (!$(this).hasClass('clicked')) {
            $(this).removeClass("shadow-depth-2");
            $(this).addClass("shadow-depth-1");
        }
    });

    $('.github-banner').mouseover(function() {
        $(this).addClass("shadow-depth-3");
    });

    $('.github-banner').mouseout(function() {
        if (!$(this).hasClass('clicked')) {
            $(this).removeClass("shadow-depth-3");
        }
    });

    $(document).click(function(event) { 
        if(!$(event.target).closest('.search-bar').length) {
            $('.search-bar').removeClass("clicked");
            $('.search-bar').removeClass("shadow-depth-2");
            $('.search-bar').addClass("shadow-depth-1");
        }        
    });
});

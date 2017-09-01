$(document).ready(function() {
    'use strict';
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
    });
    $('.view-button').on('change', function() {
        $('#sg-form').submit();
    });
});

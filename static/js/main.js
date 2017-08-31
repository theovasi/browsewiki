$(document).ready(function() {
    'use strict';

    $('.view-button').click(function() {
        $('.view-button').removeClass('active');
        $(this).addClass('active');
    });
    $('.view-button').on('change', function() {
        //$('#sg-form').submit();
    });
});

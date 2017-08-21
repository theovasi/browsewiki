$(document).ready(function() {
    'use strict';
    $('.select-button').click(function() {
        if ($(this).text() == 'Selected'){
            $(this).text('Select');
        } else {
            $(this).text('Selected');
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

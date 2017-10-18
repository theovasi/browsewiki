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
        $('input[name=cluster_select]').removeAttr('checked')
        $('.view-button').removeClass('active');
        $(this).addClass('active');
        $('#sg-form').submit();
    });
    $('.view-button').on('change', function() {
    });
});

$(document).ready(function() {
    'use strict';
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
});

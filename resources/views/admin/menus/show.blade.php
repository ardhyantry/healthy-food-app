@extends('layouts.adminSB')

@section('title', 'Quickbite - Menu Details')

@section('content')
<div class="container mt-4">
    <a class="btn btn-secondary mb-3" href="{{ url()->previous() }}">
        ← Go Back
    </a>

    <div class="card shadow">
        <div class="card-header d-flex justify-content-between align-items-center">
            <h4 class="mb-0">{{ $menu->name }}</h4>
            <!-- Button trigger modal -->
            <button type="button" class="btn btn-sm btn-primary show-image-btn" 
                    data-id="{{ $menu->id }}" 
                    data-bs-toggle="modal" 
                    data-bs-target="#imageModal">
                Show Image
            </button>
        </div>
        <div class="card-body">
            <p><strong>Nutrition Facts:</strong> {{ $menu->nutrition_fact }}</p>
            <p><strong>Category:</strong> {{ $menu->category->name }}</p>
            <p><strong>Description:</strong> {{ $menu->description }}</p>
            <p><strong>Price:</strong> Rp{{ number_format($menu->price, 0, ',', '.') }}</p>
            <p><strong>Stock:</strong> {{ $menu->stock }}</p>
        </div>
    </div>
</div>

<!-- Modal for Image -->
<div class="modal fade" id="imageModal" tabindex="-1" aria-labelledby="imageModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content shadow">
            <div class="modal-header">
                <h5 class="modal-title" id="imageModalLabel">Menu Image</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body text-center">
                <!-- Image will be loaded here -->
                <p class="text-muted">Loading image...</p>
            </div>
        </div>
    </div>
</div>
@endsection

@push('scripts')
<script>
    $(document).ready(function () {
        $('.show-image-btn').on('click', function () {
            var menuId = $(this).data('id');
            var modalBody = $('#imageModal .modal-body');
            modalBody.html('<p class="text-muted">Loading image...</p>');
            $.ajax({
                url: '/menus/' + menuId + '/image',
                method: 'GET',
                success: function (data) {
                    if (data.image_path) {
                        modalBody.html(`
                            <img src="/image/${data.image_path}" 
                                 class="img-fluid rounded shadow" 
                                 alt="Menu Image">
                        `);
                    } else {
                        modalBody.html('<p class="text-muted">No image available.</p>');
                    }
                },
                error: function () {
                    modalBody.html('<p class="text-danger">Failed to load image.</p>');
                }
            });
        });
    });
</script>
@endpush

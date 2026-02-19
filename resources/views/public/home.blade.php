@extends('layouts.custFeane')

@section('title', 'Quickbite - Home')

@section('menu_category_section')
    <section class="food_section layout_padding">
        <div class="container">
            <div class="heading_container heading_center">
                <h2>Our Menu</h2>
            </div>

            <div class="filters-content">
                <ul class="filters_menu list-inline text-center mb-4">
                    <li class="list-inline-item btn btn-dark active" data-filter="*">All</li>
                    @foreach ($categories as $category)
                        <li class="list-inline-item btn btn-outline-dark"
                            data-filter=".{{ strtolower(str_replace(' ', '-', $category->name)) }}">
                            {{ $category->name }}
                        </li>
                    @endforeach
                </ul>
            </div>
        </div>
    </section>
@endsection

@section('content')
    <div class="container">
        <div id="order-ready-alert" class="alert alert-success text-center d-none"></div>

        <div class="row grid">
            @foreach ($listMenu as $categoryName => $menus)
                @foreach ($menus as $menu)
                    <div class="col-sm-6 col-lg-4 all {{ strtolower(str_replace(' ', '-', $categoryName)) }}">
                        <div class="box shadow rounded-4 mb-4">
                            <div>
                                <div class="img-box">
                                    <img src="{{ asset('storage/' . $menu->image_path) }}" alt="{{ $menu->name }}"
                                        class="img-fluid rounded-top-4">
                                </div>
                                <div class="detail-box p-3">
                                    <h5>{{ $menu->name }}</h5>
                                    <p>{{ $menu->description }}</p>
                                    <small
                                        class="text-muted d-block mb-2">{{ $menu->nutrition_fact ?? 'No nutrition info' }}</small>
                                    <div class="options d-flex justify-content-between align-items-center">
                                        <h6 class="text-primary mb-0">Rp{{ number_format($menu->price, 0, ',', '.') }}</h6>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                @endforeach
            @endforeach
        </div>

        @section('buttonViewMore')
            <div class="btn-box text-center mt-4">
                <a href="{{ route('public.menus.index') }}">View More</a>
            </div>
        @endsection
    </div>
@endsection

@push('scripts')
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/isotope-layout@3/dist/isotope.pkgd.min.js"></script>

    <script>
        setInterval(function () {
            $.ajax({
                url: "{{ route('order.checkStatus') }}",
                type: "GET",
                success: function (response) {
                    if (response.status === 'ready') {
                        $('#order-ready-alert')
                            .removeClass('d-none')
                            .html(`<strong>✅ Pesanan Siap!</strong> ${response.message}`);
                    }
                },
                error: function (xhr) {
                    console.error('AJAX error:', xhr.responseText);
                }
            });
        }, 5000);
    </script>

    <script>
        $(document).ready(function () {
            var $grid = $('.grid').isotope({
                itemSelector: '.col-sm-6',
                layoutMode: 'fitRows'
            });

            $('.filters_menu li').on('click', function () {
                var filterValue = $(this).attr('data-filter');
                $grid.isotope({ filter: filterValue });

                $('.filters_menu li').removeClass('active');
                $(this).addClass('active');
            });
        });
    </script>
@endpush
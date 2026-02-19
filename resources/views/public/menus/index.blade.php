@extends('layouts.custFeane')

@section('title', 'Quickbite - Our Menu')


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

    @if(session('success'))
        <div class="container mt-3">
            <div class="alert alert-success alert-dismissible fade show mx-auto shadow" role="alert"
                style="max-width: 600px; position: sticky; top: 80px; z-index: 1050;">
                {{ session('success') }}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        </div>
    @endif
@endsection

@section('content')
    @foreach ($listMenu as $menu)
        <div class="col-sm-6 col-lg-4 all {{ strtolower(str_replace(' ', '-', $menu->category->name)) }}">
            <div class="box shadow rounded-4 mb-4" data-bs-toggle="modal" data-bs-target="#menuModal{{ $menu->id }}">
                <div>
                    <div class="img-box">
                        <img src="{{ asset('storage/' . $menu->image_path) }}" alt="{{ $menu->name }}"
                            class="img-fluid rounded-top-4">
                    </div>
                    <div class="detail-box p-3">
                        <h5>{{ $menu->name }}</h5>
                        <p>{{ $menu->description }}</p>
                        <small class="text-muted d-block mb-2">{{ $menu->nutrition_fact ?? 'No nutrition info' }}</small>
                        <div class="options d-flex justify-content-between align-items-center">
                            <h6 class="text-primary mb-0">Rp{{ number_format($menu->price, 0, ',', '.') }}</h6>

                            <form action="{{ route('cart.add', $menu->id) }}" method="POST">
                                @csrf
                                <button type="submit" class="btn btn-sm btn-success">
                                    <i class="fa fa-shopping-cart me-1"></i> Tambah
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="modal fade" id="menuModal{{ $menu->id }}" tabindex="-1" aria-labelledby="menuModalLabel{{ $menu->id }}"
            aria-hidden="true">
            <div class="modal-dialog modal-lg modal-dialog-centered">
                <div class="modal-content rounded-4">
                    <div class="modal-header">
                        <h5 class="modal-title" id="menuModalLabel{{ $menu->id }}">{{ $menu->name }}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body d-flex flex-column flex-md-row">
                        <div class="me-md-3 mb-3 mb-md-0 text-center">
                            <img src="{{ asset('storage/' . $menu->image_path) }}" alt="{{ $menu->name }}"
                                class="img-fluid rounded-4" style="max-height: 250px;">
                        </div>
                        <div>
                            <p>{{ $menu->description }}</p>
                            <p><strong>Nutrition:</strong> {{ $menu->nutrition_fact ?? 'No nutrition info' }}</p>
                            <p><strong>Harga:</strong> Rp{{ number_format($menu->price, 0, ',', '.') }}</p>

                            <form action="{{ route('cart.add', $menu->id) }}" method="POST">
                                @csrf
                                <button type="submit" class="btn btn-primary mt-2">
                                    <i class="fa fa-shopping-cart me-1"></i> Tambah ke Keranjang
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    @endforeach
@endsection

@push('scripts')
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/isotope-layout@3/dist/isotope.pkgd.min.js"></script>
    <script>
        $(document).ready(function () {
            var $grid = $('.grid').isotope({
                itemSelector: '.all',
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

@extends('layouts.adminSB')

@section('title', 'Quickbite - Dashboard')

@section('content')
<div class="container mt-4">
    <h2 class="mb-4">Menu Report</h2>
    <div class="row">
        <div class="col-md-3">
            <div class="card border-left-primary shadow py-2">
                <div class="card-body">
                    <h6 class="text-primary">Total Menu</h6>
                    <h3>{{ $report['total'] }}</h3>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card border-left-success shadow py-2">
                <div class="card-body">
                    <h6 class="text-success">Total Categories</h6>
                    <h3>{{ $report['categories'] }}</h3>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card border-left-danger shadow py-2">
                <div class="card-body">
                    <h6 class="text-danger">Most Expensive</h6>
                    <p class="mb-0">{{ $report['most_expensive']?->name ?? '-' }}</p>
                    <strong>Rp{{ number_format($report['most_expensive']?->price ?? 0, 0, ',', '.') }}</strong>
                </div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card border-left-info shadow py-2">
                <div class="card-body">
                    <h6 class="text-info">Cheapest</h6>
                    <p class="mb-0">{{ $report['cheapest']?->name ?? '-' }}</p>
                    <strong>Rp{{ number_format($report['cheapest']?->price ?? 0, 0, ',', '.') }}</strong>
                </div>
            </div>
        </div>
    </div>

    <div class="row mt-4">
        <div class="col-md-3">
            <div class="card border-left-warning shadow py-2">
                <div class="card-body">
                    <h6 class="text-warning">Average Price</h6>
                    <h3>Rp{{ number_format($report['average_price'], 0, ',', '.') }}</h3>
                </div>
            </div>
        </div>
    </div>

    <hr class="my-4">

</div>


@endsection